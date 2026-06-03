"""
core/agent/ppo.py
────────────────────────────────────────────────────────────────────────────
PPOAgent — the SINGLE live agent (DESIGN_DECISIONS.md #1). Actor-critic PPO with
three policy heads over a shared trunk:

  - direction head : Categorical(DIRECTION_DIM=3)   FLAT / BUY / SELL
  - exit head      : Categorical(EXIT_DIM=3)         HOLD / REDUCE / CLOSE
  - lot head       : Gaussian -> sigmoid -> [0,1]    continuous lot fraction
  - value head     : scalar V(s)

Training: on-policy rollouts -> GAE(λ) advantages -> clipped surrogate objective
with entropy bonus and value loss. No replay buffer, no epsilon, no target net
(those were DQN concepts and are gone).

Masking (DESIGN_DECISIONS.md #2): a per-step direction mask of shape
(B, DIRECTION_DIM) zeroes disallowed directions BEFORE the categorical sample.
The code NEVER chooses BUY vs SELL — under a force_in_and_gate phase we only mask
FLAT (so the agent must open something), leaving BUY and SELL both available.

Device-agnostic: AMP/torch.compile only engage on CUDA; identical logic on CPU.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from core.agent.action_space import DIRECTION_DIM, EXIT_DIM, FLAT
from core.env.environment import OBS_SCHEMA_VERSION

_NEG_INF = -1e9


class ActorCritic(nn.Module):
    """Shared MLP trunk + direction/exit/lot/value heads."""

    def __init__(self, state_dim: int, hidden: int = 256,
                 lot_log_std_init: float = -0.5):
        super().__init__()
        self.state_dim = state_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.dir_head = nn.Linear(hidden, DIRECTION_DIM)
        self.exit_head = nn.Linear(hidden, EXIT_DIM)
        self.lot_mean = nn.Linear(hidden, 1)
        # state-independent log-std for the continuous lot head. Initialized to a
        # modest exploratory value (exp(-0.5)≈0.61) and FLOORED in PPOAgent._dists
        # so the sizing head never collapses to a deterministic 0-variance lot
        # (a collapse symptom of the old do-nothing ~$0 policy).
        self.lot_log_std = nn.Parameter(torch.full((1,), float(lot_log_std_init)))
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        # .clone() returns fresh storage for each head output so nothing the
        # rollout buffer keeps a reference to can be overwritten in-place by a
        # later compiled forward pass. Combined with torch.compile(mode="default")
        # (NOT reduce-overhead / CUDA Graphs) this fixes the rollout-buffer
        # corruption that crashed torch.stack() in update().
        return (self.dir_head(h).clone(), self.exit_head(h).clone(),
                self.lot_mean(h).clone(), self.value_head(h).squeeze(-1).clone())


class RolloutBuffer:
    """Stores on-policy transitions for one PPO update, then clears."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.dir_a, self.exit_a, self.lot_a = [], [], [], []
        self.logp, self.rewards, self.dones, self.values = [], [], [], []
        self.dir_mask = []

    def add(self, state, dir_a, exit_a, lot_a, logp, reward, done, value, dir_mask):
        self.states.append(state)
        self.dir_a.append(dir_a)
        self.exit_a.append(exit_a)
        self.lot_a.append(lot_a)
        self.logp.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.dir_mask.append(dir_mask)

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """Pure-PPO trading agent. Replaces the deprecated DQNAgent everywhere."""

    def __init__(self, state_dim: int, cfg: dict, device: torch.device):
        self.state_dim = state_dim
        self.cfg = cfg
        self.device = device
        ppo = cfg.get("PPO", {}) or {}
        self.gamma = float(ppo.get("gamma", cfg.get("GAMMA", 0.95)))
        self.gae_lambda = float(ppo.get("gae_lambda", 0.95))
        self.clip = float(ppo.get("clip_range", 0.2))
        self.ent_coef = float(ppo.get("ent_coef", 0.01))
        self.vf_coef = float(ppo.get("vf_coef", 0.5))
        self.epochs = int(ppo.get("n_epochs", 4))
        self.max_grad_norm = float(ppo.get("max_grad_norm", 0.5))
        self.lr = float(ppo.get("learning_rate", cfg.get("LR", 3e-4)))

        self.lot_log_std_min = float(ppo.get("lot_log_std_min", -2.0))
        hidden = int(cfg.get("HIDDEN", 256))
        self.net = ActorCritic(
            state_dim, hidden,
            lot_log_std_init=float(ppo.get("lot_log_std_init", -0.5))).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.buffer = RolloutBuffer()

        # ── PROPORTIONAL-SCALER BASELINE (target_aware_policy.md item 6) ──────
        # The target/DD the policy was trained at. Defaults to the cfg fallback
        # (0.025/0.01) and is OVERWRITTEN by whatever a loaded checkpoint persisted
        # (its training-time midpoint when --randomize-ftmo was used). The item-6
        # scaler reads these at inference; see core/env/environment.proportional_
        # lot_scale and training/estimate_pass_prob.py.
        self.trained_target_pct = float(cfg.get("TRAINED_TARGET_PCT",
                                        cfg.get("DAILY_TARGET_PCT", 0.025)))
        self.trained_max_dd_pct = float(cfg.get("TRAINED_MAX_DD_PCT",
                                        cfg.get("DAILY_MAX_DD_PCT", 0.010)))

        self.use_amp = bool(cfg.get("USE_AMP", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._fwd = self.net
        if bool(cfg.get("USE_TORCH_COMPILE", True)) and device.type == "cuda":
            try:
                # mode="default" compiles without CUDA Graphs. "reduce-overhead"
                # reuses the same memory buffer every forward pass, overwriting
                # rollout buffer tensors mid-episode and crashing torch.stack()
                # in update(). "default" still gives a significant A100 speedup.
                self._fwd = torch.compile(self.net, mode="default")
            except Exception:                                  # pragma: no cover
                self._fwd = self.net

    # ── helpers ──────────────────────────────────────────────────────────────
    def _dists(self, dir_logits, exit_logits, lot_mean,
               dir_mask: Optional[torch.Tensor]):
        if dir_mask is not None:
            dir_logits = dir_logits + (1.0 - dir_mask) * _NEG_INF
        dir_d = Categorical(logits=dir_logits)
        exit_d = Categorical(logits=exit_logits)
        # Floor the log-std so the continuous lot head keeps exploring (never a
        # deterministic 0-variance lot). clamp(min=...) is differentiable above
        # the floor and a no-op gradient at it.
        log_std = self.net.lot_log_std.clamp(min=self.lot_log_std_min)
        std = torch.exp(log_std).expand_as(lot_mean)
        lot_d = Normal(lot_mean.squeeze(-1), std.squeeze(-1))
        return dir_d, exit_d, lot_d

    # ── action selection (rollout) ──────────────────────────────────────────
    @torch.no_grad()
    def select_actions(self, state: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Batched action sampling. state (B,state_dim). mask (B,DIRECTION_DIM) zeroes
        disallowed directions. Returns dict of tensors: direction, exit, lot_raw,
        logp, value. Lot is sigmoid-squashed to [0,1].
        """
        dir_logits, exit_logits, lot_mean, value = self._fwd(state)
        dir_d, exit_d, lot_d = self._dists(dir_logits, exit_logits, lot_mean, mask)
        dir_a = dir_d.sample()
        exit_a = exit_d.sample()
        lot_pre = lot_d.sample()
        lot_raw = torch.sigmoid(lot_pre)
        logp = dir_d.log_prob(dir_a) + exit_d.log_prob(exit_a) + lot_d.log_prob(lot_pre)
        return {"direction": dir_a, "exit": exit_a, "lot_raw": lot_raw,
                "lot_pre": lot_pre, "logp": logp, "value": value}

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, deterministic: bool = True,
                      mask: Optional[torch.Tensor] = None) -> Tuple[int, float, int]:
        """Single-obs inference (live_runner / policy_inspector). Returns
        (direction, lot_raw, exit). Deterministic = argmax dir/exit, mean lot."""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        dir_logits, exit_logits, lot_mean, _v = self._fwd(obs)
        m = mask.reshape(1, -1) if mask is not None else None
        dir_d, exit_d, _lot_d = self._dists(dir_logits, exit_logits, lot_mean, m)
        if deterministic:
            direction = int(dir_d.probs.argmax(dim=-1).item())
            exit_a = int(exit_d.probs.argmax(dim=-1).item())
            lot_raw = float(torch.sigmoid(lot_mean.squeeze()).item())
        else:
            direction = int(dir_d.sample().item())
            exit_a = int(exit_d.sample().item())
            lot_raw = float(torch.sigmoid(Normal(
                lot_mean.squeeze(-1), torch.exp(self.net.lot_log_std)).sample()).item())
        return direction, lot_raw, exit_a

    # ── proportional lot scaler (item 6) ──────────────────────────────────────
    def proportional_scale(self, current_target_pct: float,
                           current_max_dd_pct: float) -> float:
        """The bounded, deterministic effective_lot_scale for the CURRENT
        target/DD vs the policy's TRAINED baseline (target_aware_policy.md item 6).
        1.0 at baseline; tighter DD scales DOWN, higher target scales UP; always
        within [PROPORTIONAL_SCALE_LO, PROPORTIONAL_SCALE_HI]. Returns 1.0 when the
        scaler is toggled OFF (CFG['PROPORTIONAL_SCALER'] = False)."""
        from core.env.environment import proportional_lot_scale
        if not bool(self.cfg.get("PROPORTIONAL_SCALER", True)):
            return 1.0
        return proportional_lot_scale(
            current_target_pct, current_max_dd_pct,
            self.trained_target_pct, self.trained_max_dd_pct,
            lo=float(self.cfg.get("PROPORTIONAL_SCALE_LO", 0.25)),
            hi=float(self.cfg.get("PROPORTIONAL_SCALE_HI", 2.0)))

    @torch.no_grad()
    def select_actions_eval(self, state: torch.Tensor,
                            mask: Optional[torch.Tensor] = None,
                            lot_scale: float = 1.0) -> Dict[str, torch.Tensor]:
        """Batched DETERMINISTIC action selection for eval/estimation: argmax
        direction/exit and the MEAN lot (sigmoid of the lot-head mean), with the
        item-6 proportional `lot_scale` applied ON TOP of the chosen lot (clamped
        back into [0,1]). Returns the same dict shape env.step() expects. The
        scaler only resizes exposure — direction/exit remain the policy's choice."""
        dir_logits, exit_logits, lot_mean, value = self._fwd(state)
        dir_d, exit_d, _lot_d = self._dists(dir_logits, exit_logits, lot_mean, mask)
        dir_a = dir_d.probs.argmax(dim=-1)
        exit_a = exit_d.probs.argmax(dim=-1)
        lot_raw = torch.sigmoid(lot_mean.squeeze(-1))
        lot_raw = (lot_raw * float(lot_scale)).clamp(0.0, 1.0)
        return {"direction": dir_a, "exit": exit_a, "lot_raw": lot_raw,
                "value": value}

    # ── rollout storage ───────────────────────────────────────────────────────
    def store(self, state, out: dict, reward, done, dir_mask):
        self.buffer.add(state.detach(), out["direction"].detach(),
                        out["exit"].detach(), out["lot_pre"].detach(),
                        out["logp"].detach(), reward.detach(), done.detach(),
                        out["value"].detach(),
                        dir_mask.detach() if dir_mask is not None else None)

    # ── PPO update ─────────────────────────────────────────────────────────────
    def update(self, last_value: Optional[torch.Tensor] = None) -> Optional[float]:
        """Run a PPO update over the collected rollout, then clear the buffer.
        Returns the mean total loss (or None if no data)."""
        if len(self.buffer) == 0:
            return None
        states = torch.stack(self.buffer.states)          # (T,B,state_dim)
        dir_a = torch.stack(self.buffer.dir_a)
        exit_a = torch.stack(self.buffer.exit_a)
        lot_pre = torch.stack(self.buffer.lot_a)
        old_logp = torch.stack(self.buffer.logp)
        rewards = torch.stack(self.buffer.rewards)
        dones = torch.stack(self.buffer.dones).float()
        values = torch.stack(self.buffer.values)
        masks = self.buffer.dir_mask

        T = states.shape[0]
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(rewards[0])
        next_value = last_value if last_value is not None else torch.zeros_like(values[0])
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
            next_value = values[t]
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # flatten time*batch
        def flat(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)
        s_f = states.reshape(-1, states.shape[-1])
        dir_f, exit_f, lotpre_f = flat(dir_a), flat(exit_a), flat(lot_pre)
        oldlogp_f, adv_f, ret_f = flat(old_logp), flat(adv), flat(returns)
        mask_f = (torch.stack(masks).reshape(-1, masks[0].shape[-1])
                  if masks and masks[0] is not None else None)

        total = 0.0
        for _ in range(self.epochs):
            dl, el, lm, v = self.net(s_f)
            dd, ed, ld = self._dists(dl, el, lm, mask_f)
            new_logp = dd.log_prob(dir_f) + ed.log_prob(exit_f) + ld.log_prob(lotpre_f)
            ratio = torch.exp(new_logp - oldlogp_f)
            s1 = ratio * adv_f
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_f
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = F.mse_loss(v, ret_f)
            entropy = (dd.entropy().mean() + ed.entropy().mean() + ld.entropy().mean())
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total += float(loss.item())
        self.buffer.clear()
        return total / self.epochs

    # ── checkpoint I/O (PPO only) ──────────────────────────────────────────────
    def save(self, path: str, extra: dict = None):
        # obs_schema_version (target_aware_policy.md item 4/6) lets a resume detect
        # an input-layer width mismatch instead of silently mis-loading. The
        # trained target/DD BASELINE is also persisted so the item-6 proportional
        # scaler knows what regime the policy learned at (defaults 0.025/0.01; the
        # MIDPOINT of the randomization ranges when --randomize-ftmo was used).
        ppo = self.cfg.get("PPO", {}) or {}
        if bool(self.cfg.get("RANDOMIZE_FTMO_INPUTS", False)):
            tlo, thi = self.cfg.get("RANDOMIZE_TARGET_RANGE", [0.01, 0.05])
            dlo, dhi = self.cfg.get("RANDOMIZE_DD_RANGE", [0.005, 0.02])
            trained_target = 0.5 * (float(tlo) + float(thi))
            trained_dd = 0.5 * (float(dlo) + float(dhi))
        else:
            trained_target = float(self.cfg.get("TRAINED_TARGET_PCT",
                                   self.cfg.get("DAILY_TARGET_PCT", 0.025)))
            trained_dd = float(self.cfg.get("TRAINED_MAX_DD_PCT",
                               self.cfg.get("DAILY_MAX_DD_PCT", 0.010)))
        payload = {"net": self.net.state_dict(),
                   "optimizer": self.optimizer.state_dict(),
                   "state_dim": self.state_dim, "agent": "ppo",
                   "obs_schema_version": OBS_SCHEMA_VERSION,
                   "trained_target_pct": trained_target,
                   "trained_max_dd_pct": trained_dd}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str, partial: bool = False) -> dict:
        """Load a checkpoint. Detects an OBSERVATION-SCHEMA mismatch (the input
        layer width / obs_schema_version differs from THIS agent's) and handles it
        CLEANLY rather than silently loading a mismatched net (target_aware_policy
        .md item 4): the trunk's input layer (trunk.0.*) is reinitialized fresh
        while every other layer that still matches is loaded, with a LOUD log. This
        is the documented behaviour for the v1->v2 obs bump (7 new target/risk
        features). A full match loads normally."""
        # weights_only=False: our checkpoints store metadata dicts (trusted, ours).
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Restore the trained target/DD baseline for the item-6 proportional scaler
        # (falls back to the current values if the checkpoint predates the field).
        if ckpt.get("trained_target_pct") is not None:
            self.trained_target_pct = float(ckpt["trained_target_pct"])
        if ckpt.get("trained_max_dd_pct") is not None:
            self.trained_max_dd_pct = float(ckpt["trained_max_dd_pct"])
        sd = ckpt.get("net", {})
        ckpt_schema = ckpt.get("obs_schema_version")
        ckpt_state_dim = ckpt.get("state_dim")
        in_w = sd.get("trunk.0.weight")
        ckpt_in_dim = int(in_w.shape[1]) if in_w is not None else ckpt_state_dim
        schema_mismatch = (
            (ckpt_schema is not None and ckpt_schema != OBS_SCHEMA_VERSION)
            or (ckpt_in_dim is not None and int(ckpt_in_dim) != int(self.state_dim))
        )
        if schema_mismatch:
            print("─" * 70, flush=True)
            print("[ppo] ⚠️  OBSERVATION-SCHEMA MISMATCH on resume:", flush=True)
            print(f"      checkpoint obs_schema_version={ckpt_schema} "
                  f"(input dim {ckpt_in_dim}) vs current v{OBS_SCHEMA_VERSION} "
                  f"(input dim {self.state_dim}).", flush=True)
            print("      The observation layout changed (target/risk-aware "
                  "features were added).", flush=True)
            print("      → Reinitializing JUST the input layer (trunk.0.*) fresh "
                  "and loading every other matching layer.", flush=True)
            print("      Training continues from these partially-transferred "
                  "weights; the new input layer learns the added features.",
                  flush=True)
            print("─" * 70, flush=True)
            own = self.net.state_dict()
            for k, v in sd.items():
                # Skip the input layer on a mismatch (its width changed); load the
                # rest only where the shape still matches exactly.
                if k.startswith("trunk.0."):
                    continue
                if k in own and own[k].shape == v.shape:
                    own[k] = v
            self.net.load_state_dict(own)
            return ckpt
        if partial:
            own = self.net.state_dict()
            for k, v in sd.items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v
            self.net.load_state_dict(own)
        else:
            self.net.load_state_dict(sd)
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:                                  # pragma: no cover
                pass
        return ckpt
