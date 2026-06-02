"""
core/agent/dqn.py
────────────────────────────────────────────────────────────────────────────
GPU DQN agent. Ported from gpu_rl_trading/agent/dqn.py + replay.py (REPO1) with:

  (a) NUM_ACTIONS = 756 imported from action_space.py (never hardcoded).
  (b) torch.compile(policy_net, mode="reduce-overhead") on CUDA (PyTorch 2.x).
  (c) torch.amp.autocast("cuda") + GradScaler for AMP forward/backward on CUDA.
  (d) BATCH_SIZE_RL / MEMORY_SIZE taken from cfg (A100: 2048 / 500_000).
  (e) Transfer learning: load_partial() re-initializes the OUTPUT layer (Kaiming)
      when a checkpoint's action dim != 756, copying all hidden layers and the
      input layer. Used to bridge the old 7-action checkpoint to the new space.

All compile/AMP paths are guarded so the SAME code runs on CPU (dev/CI) without
GPU features — only speed differs.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.agent.action_space import NUM_ACTIONS


# ── GPU replay buffer (ported from replay.py) ────────────────────────────────
class GPUReplayBuffer:
    """Transitions stored directly in device tensors for fast sampling."""

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((self.capacity, state_dim), device=device)
        self.next_states = torch.zeros((self.capacity, state_dim), device=device)
        self.actions = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(self.capacity, device=device)
        self.dones = torch.zeros(self.capacity, dtype=torch.bool, device=device)

    def push(self, state, action, reward, next_state, done):
        B = state.shape[0]
        idx = torch.arange(self.ptr, self.ptr + B, device=self.device) % self.capacity
        self.states[idx] = state.detach()
        self.next_states[idx] = next_state.detach()
        self.actions[idx] = action.detach().long()
        self.rewards[idx] = reward.detach().float()
        self.dones[idx] = done.detach().bool()
        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size


class QNetwork(nn.Module):
    """MLP Q-network: state_dim -> hidden -> hidden/2 -> num_actions."""

    def __init__(self, state_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_partial(self, state_dict: dict, old_num_actions: int):
        """
        Transfer-learning load. Copies input + hidden layers; re-inits the output
        layer (Kaiming uniform) when the action dim changed (e.g. 7 -> 756).
        """
        own = self.state_dict()
        for name, param in state_dict.items():
            if name not in own:
                continue
            if name == "net.4.weight" or name == "net.4.bias":
                # output layer — only copy if shapes match; else keep fresh init
                if own[name].shape == param.shape:
                    own[name] = param
                else:
                    print(f"[transfer] re-init output layer {name}: "
                          f"{tuple(param.shape)} -> {tuple(own[name].shape)}",
                          flush=True)
            elif own[name].shape == param.shape:
                own[name] = param
        self.load_state_dict(own)
        # ensure the (possibly fresh) output layer is properly initialized
        out = self.net[4]
        if out.out_features != old_num_actions:
            nn.init.kaiming_uniform_(out.weight, a=5 ** 0.5)
            nn.init.zeros_(out.bias)


class DQNAgent:
    """Epsilon-greedy DQN with target net, AMP, torch.compile, transfer learning."""

    def __init__(self, state_dim: int, num_actions: int, cfg: dict,
                 device: torch.device):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.cfg = cfg
        self.device = device
        self.gamma = float(cfg.get("GAMMA", 0.95))
        self.epsilon = float(cfg.get("EPSILON_START", 0.9))
        self.eps_min = float(cfg.get("EPSILON_MIN", 0.05))
        self.batch_size = int(cfg.get("BATCH_SIZE_RL", 2048))
        self.train_every = int(cfg.get("TRAIN_EVERY", 2))
        self.sync_every = int(cfg.get("SYNC_EVERY", 200))
        self._step_count = 0

        hidden = int(cfg.get("HIDDEN", 256))
        self.policy_net = QNetwork(state_dim, num_actions, hidden).to(device)
        self.target_net = QNetwork(state_dim, num_actions, hidden).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=float(cfg.get("LR", 5e-4)))
        self.memory = GPUReplayBuffer(int(cfg.get("MEMORY_SIZE", 500_000)),
                                      state_dim, device)

        # AMP only on CUDA
        self.use_amp = bool(cfg.get("USE_AMP", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # torch.compile only on CUDA (and PyTorch 2.x); guarded against failure
        self._forward = self.policy_net
        if bool(cfg.get("USE_TORCH_COMPILE", True)) and device.type == "cuda":
            try:
                self._forward = torch.compile(self.policy_net,
                                              mode="reduce-overhead")
                print("[dqn] torch.compile enabled (reduce-overhead)", flush=True)
            except Exception as exc:                       # pragma: no cover
                print(f"[dqn] torch.compile unavailable ({exc}); using eager",
                      flush=True)

    # ── action selection ──────────────────────────────────────────────────────
    @torch.no_grad()
    def _q(self, state: torch.Tensor) -> torch.Tensor:
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                return self._forward(state)
        return self._forward(state)

    @torch.no_grad()
    def select_actions(self, state: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Batched epsilon-greedy. state (B,state_dim) -> actions (B,). Optional
        (B,num_actions) mask adds -1e9 to disallowed actions before argmax."""
        B = state.shape[0]
        q = self._q(state).float()
        if mask is not None:
            q = q + (1.0 - mask) * (-1e9)
        explore = torch.rand(B, device=self.device) < self.epsilon
        greedy = q.argmax(dim=1)
        rand = torch.randint(0, self.num_actions, (B,), device=self.device)
        return torch.where(explore, rand, greedy)

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, deterministic: bool = False,
                      mask: Optional[torch.Tensor] = None) -> int:
        """Single-observation inference (used by live_runner / policy_inspector)."""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        q = self._q(obs).float()
        if mask is not None:
            q = q + (1.0 - mask.reshape(1, -1)) * (-1e9)
        if not deterministic and torch.rand(1).item() < self.epsilon:
            return int(torch.randint(0, self.num_actions, (1,)).item())
        return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        self._step_count += 1
        if self._step_count % self.train_every != 0:
            return None
        s, a, r, ns, d = self.memory.sample(self.batch_size)
        with torch.no_grad():
            q_next = self.target_net(ns).max(dim=1).values
            targets = r + self.gamma * q_next * (1.0 - d.float())
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                q_pred = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q_pred, targets)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_pred = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q_pred, targets)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
        if self._step_count % self.sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def decay_epsilon(self, episode: int):
        decay = self.cfg.get("EPSILON_DECAY_EPISODES", 500)
        start = float(self.cfg.get("EPSILON_START", 0.9))
        ratio = self.eps_min / (start + 1e-8)
        self.epsilon = max(self.eps_min, start * (ratio ** (episode / decay)))

    # ── checkpoint I/O ──────────────────────────────────────────────────────────
    def save(self, path: str, extra: dict = None):
        payload = {
            "q_net": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step": self._step_count,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
        }
        # BUG #8 FIX: persist the replay buffer so a crash-resume restores the
        # agent's experience (not just weights). Stored as CPU tensors so the
        # checkpoint survives device changes (A100 <-> CPU). Skipped when the
        # buffer is tiny (inference contexts use MEMORY_SIZE=1).
        if self.memory.size > 1:
            payload.update({
                "replay_states": self.memory.states[:self.memory.size].cpu(),
                "replay_next_states": self.memory.next_states[:self.memory.size].cpu(),
                "replay_actions": self.memory.actions[:self.memory.size].cpu(),
                "replay_rewards": self.memory.rewards[:self.memory.size].cpu(),
                "replay_dones": self.memory.dones[:self.memory.size].cpu(),
                "replay_size": self.memory.size,
                "replay_ptr": self.memory.ptr,
            })
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str, partial: bool = False) -> dict:
        # weights_only=False is REQUIRED here: our checkpoints intentionally
        # store non-tensor metadata (phase, phi, episode, nested dicts). Under
        # PyTorch 2.6+ the default weights_only=True would raise UnpicklingError
        # on this payload. These are our own trusted checkpoints, so disabling
        # the restriction is safe (do NOT load untrusted .pt files with this).
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_actions = ckpt.get("num_actions", self.num_actions)
        if partial and ckpt_actions != self.num_actions:
            print(f"[transfer] num_actions {ckpt_actions} -> {self.num_actions}; "
                  f"re-init output layer, epsilon={self.cfg.get('TRANSFER_EPSILON',0.3)}",
                  flush=True)
            self.policy_net.load_partial(ckpt["q_net"], ckpt_actions)
            self.target_net.load_partial(ckpt["target"], ckpt_actions)
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                              lr=float(self.cfg.get("LR", 5e-4)))
            self.epsilon = float(self.cfg.get("TRANSFER_EPSILON", 0.3))
        else:
            self.policy_net.load_state_dict(ckpt["q_net"])
            self.target_net.load_state_dict(ckpt["target"])
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:                              # pragma: no cover
                pass
            self.epsilon = ckpt.get("epsilon", self.epsilon)
        self._step_count = ckpt.get("step", 0)

        # BUG #8 FIX: restore the replay buffer when present and compatible.
        # Skip on transfer (action/state dim changed) or when sizes are
        # incompatible — weights are still loaded; the buffer just starts fresh.
        if "replay_states" in ckpt:
            size = int(ckpt["replay_size"])
            ckpt_sdim = ckpt["replay_states"].shape[1]
            if ckpt_sdim != self.state_dim or size > self.memory.capacity:
                print(f"[ckpt] replay incompatible (sdim {ckpt_sdim} vs "
                      f"{self.state_dim}, size {size} vs cap {self.memory.capacity})"
                      f" — starting fresh buffer", flush=True)
            else:
                self.memory.states[:size] = ckpt["replay_states"].to(self.device)
                self.memory.next_states[:size] = ckpt["replay_next_states"].to(self.device)
                self.memory.actions[:size] = ckpt["replay_actions"].to(self.device)
                self.memory.rewards[:size] = ckpt["replay_rewards"].to(self.device)
                self.memory.dones[:size] = ckpt["replay_dones"].to(self.device)
                self.memory.size = size
                self.memory.ptr = int(ckpt["replay_ptr"]) % self.memory.capacity
                print(f"[ckpt] replay buffer restored ({size} transitions)", flush=True)
        return ckpt
