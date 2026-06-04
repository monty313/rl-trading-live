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

# fp16-safe masking penalty. -1e9 overflows the fp16 range under AMP autocast and
# can produce NaN in softmax/entropy; -1e8 still drives the masked class probability
# below 1e-30 (effectively zero) while staying numerically stable.
_NEG_INF = -1e8


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
        # ── LOT HEAD INITIALIZATION (issue-2 lot-sizing fix) ──────────────────
        # The lot head emits a pre-squash mean that select_actions turns into a lot
        # via sigmoid(lot_pre) -> [0,1] -> map_lot -> [MIN_LOT, max_lot]. We want
        # the INITIAL mean lot to sit ~MID-RANGE (~1.0 lot on a 2.0 max), NOT pinned
        # at the 0.01 floor — otherwise the agent starts effectively flat-sized and
        # has to climb out of a saturated sigmoid tail to ever size up. With the
        # default Linear init the bias is a small RANDOM value, so the mid-range
        # start was only incidental. We make it ROBUST by construction: zero the
        # bias so sigmoid(0)=0.5 (=> ~1.0 lot at MAX_LOT=2.0) and keep the weights
        # small (near state-independent at t=0) so the head starts centered and
        # LEARNS to differentiate size from there. This guarantees the documented
        # "initial mean lot is mid-range" invariant regardless of trunk init.
        nn.init.zeros_(self.lot_mean.bias)
        nn.init.normal_(self.lot_mean.weight, mean=0.0, std=0.01)
        # state-independent log-std for the continuous lot head. Initialized to a
        # modest exploratory value (exp(-0.5)≈0.61) and FLOORED in PPOAgent._dists
        # so the sizing head never collapses to a deterministic 0-variance lot
        # (a collapse symptom of the old do-nothing ~$0 policy). The floor lets PPO
        # keep SAMPLING a spread of sizes (incl. toward 2.0) so size stays learnable
        # even if the mean drifts; see PPOAgent.lot_log_std_min.
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
        # ── Section 9 — ENTROPY ANNEALING (high exploration -> stable ent_coef) ──
        # The STABLE coefficient is the PPO ent_coef above; the LIVE coefficient
        # (self.ent_coef, used in update()) starts at ENTROPY_START_COEF and is
        # annealed linearly down to it by ENTROPY_ANNEAL_EPISODES via
        # anneal_entropy(episode). All values are config-driven (nothing hardcoded).
        self.ent_coef_stable = self.ent_coef
        self._ent_anneal_on = bool(cfg.get("ENTROPY_ANNEAL_ENABLED", True))
        self._ent_start = float(cfg.get("ENTROPY_START_COEF", self.ent_coef))
        self._ent_anneal_eps = max(1, int(cfg.get("ENTROPY_ANNEAL_EPISODES", 20)))
        # Section 9 (S11): the anneal SHAPE. SMOOTH by default — "cosine" (half-cosine
        # ease, slow at both ends) or "exp" (geometric decay). "linear" is kept for
        # back-compat. A step schedule is deliberately NOT offered: a discontinuous
        # entropy jump destabilizes the policy gradient. Config-driven, nothing hardcoded.
        self._ent_shape = str(cfg.get("ENTROPY_ANNEAL_SHAPE", "cosine")).lower()
        if self._ent_anneal_on:
            self.ent_coef = self._ent_start          # begin high at episode 0
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

    def anneal_entropy(self, episode: int) -> float:
        """Section 9 (S11): SMOOTHLY anneal the LIVE entropy coefficient from
        ENTROPY_START_COEF (episode 0) down to the stable PPO ent_coef by episode
        ENTROPY_ANNEAL_EPISODES, then hold it EXACTLY at the stable value (no
        residual perturbation). The schedule shape is ENTROPY_ANNEAL_SHAPE:
          • "cosine" (default) — half-cosine ease: c = stable + (start-stable)*0.5*
            (1+cos(pi*frac)); slow near both endpoints, smooth everywhere.
          • "exp" — geometric decay: c = start*(stable/start)**frac (start,stable>0).
          • "linear" — straight interpolation (legacy).
        No step schedule (a discontinuous jump destabilizes the gradient). No-op
        (returns the stable coef) when annealing is disabled. Call once per episode.
        Returns the coefficient now in force."""
        if not self._ent_anneal_on:
            self.ent_coef = self.ent_coef_stable
            return self.ent_coef
        frac = min(1.0, max(0.0, episode / self._ent_anneal_eps))
        start, stable = self._ent_start, self.ent_coef_stable
        if self._ent_shape == "cosine":
            import math
            self.ent_coef = stable + (start - stable) * 0.5 * (1.0 + math.cos(math.pi * frac))
        elif self._ent_shape == "exp" and start > 0 and stable > 0:
            self.ent_coef = start * (stable / start) ** frac
        else:                                   # "linear" / fallback
            self.ent_coef = start + frac * (stable - start)
        return self.ent_coef

    # ── helpers ──────────────────────────────────────────────────────────────
    def _dists(self, dir_logits, exit_logits, lot_mean,
               dir_mask: Optional[torch.Tensor]):
        # ── NaN/Inf GUARD (numerical-stability fix) ───────────────────────────
        # A blown-up update() can leave NaN/Inf in the head outputs; sanitize
        # before building the distributions so Categorical's validate_args check
        # never crashes mid-train. nan_to_num maps NaN->0, +Inf->large, -Inf->-large.
        dir_logits = torch.nan_to_num(dir_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        exit_logits = torch.nan_to_num(exit_logits, nan=0.0, posinf=30.0, neginf=-30.0)
        lot_mean = torch.nan_to_num(lot_mean, nan=0.0, posinf=30.0, neginf=-30.0)
        if dir_mask is not None:
            # Use a finite, fp16-safe penalty (NOT -1e9, which underflows under AMP
            # and can yield NaN in softmax/entropy). -1e8 still zeros the masked
            # class to <1e-30 probability while staying numerically well-behaved.
            dir_logits = dir_logits + (1.0 - dir_mask) * _NEG_INF
            # Guard against a fully-masked row (all directions disallowed): give it
            # a uniform-ish fallback so Categorical gets a valid (non -inf) row.
            all_masked = (dir_mask.sum(dim=-1, keepdim=True) == 0)
            if bool(all_masked.any()):
                dir_logits = torch.where(all_masked, torch.zeros_like(dir_logits),
                                         dir_logits)
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

    # ── truncation bootstrap value ────────────────────────────────────────────
    @torch.no_grad()
    def bootstrap_value(self, state: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """V(s_T) for the state at the END of a rollout, used to bootstrap GAE on a
        TRUNCATED (time-limit) boundary. The mask is irrelevant to the value head
        (it only gates the direction logits) but is accepted for call-site symmetry.
        Returns a (B,) tensor on self.device. Detached/no-grad — it is a target."""
        _dl, _el, _lm, value = self._fwd(state)
        return value.detach()

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
        # ── NUMERICAL STABILITY: clamp normalized advantages ─────────────────
        # With raw rewards in the millions a single outlier advantage can dominate
        # the surrogate and blow up the ratio*adv term. ±10 sigma is plenty of
        # signal while bounding the gradient.
        adv = torch.clamp(adv, -10.0, 10.0)

        # flatten time*batch
        def flat(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)
        s_f = states.reshape(-1, states.shape[-1])
        dir_f, exit_f, lotpre_f = flat(dir_a), flat(exit_a), flat(lot_pre)
        oldlogp_f, adv_f, ret_f = flat(old_logp), flat(adv), flat(returns)
        mask_f = (torch.stack(masks).reshape(-1, masks[0].shape[-1])
                  if masks and masks[0] is not None else None)

        total = 0.0
        n_done = 0
        for _ in range(self.epochs):
            self.optimizer.zero_grad(set_to_none=True)
            # AMP autocast: matches the scaler created in __init__. On CPU this is a
            # no-op (enabled=False), so the logic is identical device-agnostically.
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                dl, el, lm, v = self.net(s_f)
                dd, ed, ld = self._dists(dl, el, lm, mask_f)
                new_logp = (dd.log_prob(dir_f) + ed.log_prob(exit_f)
                            + ld.log_prob(lotpre_f))
                # Clamp the log-ratio BEFORE exp() so a stale old_logp can't produce
                # an astronomically large ratio (the classic PPO overflow path).
                logratio = torch.clamp(new_logp - oldlogp_f, -20.0, 20.0)
                ratio = torch.exp(logratio)
                s1 = ratio * adv_f
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_f
                policy_loss = -torch.min(s1, s2).mean()
                # Huber (smooth_l1) instead of MSE: linear (not quadratic) for large
                # residuals, so huge returns can't square into a 1e12+ value loss
                # that dominates and explodes the total. Also clamp as a hard cap.
                value_loss = F.smooth_l1_loss(v, ret_f)
                entropy = (dd.entropy().mean() + ed.entropy().mean()
                           + ld.entropy().mean())
                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * entropy)
            # ── SKIP non-finite batches instead of corrupting the weights ─────
            # If the loss is NaN/Inf, stepping the optimizer writes NaNs into every
            # parameter and the NEXT forward pass emits nan logits -> Categorical
            # crash. Skipping keeps the last-good weights and lets training recover.
            if not torch.isfinite(loss):
                print("[ppo] ⚠️  non-finite loss in update epoch — skipping this "
                      "PPO step to protect the weights", flush=True)
                continue
            # Scaler path: scale -> backward -> unscale_ -> clip -> step -> update.
            # unscale_ MUST run before clip_grad_norm_ so clipping sees the true
            # (unscaled) gradient magnitude. The scaler also auto-skips the step if
            # it detects inf/nan grads, a second layer of protection.
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.item())
            n_done += 1
        self.buffer.clear()
        return (total / n_done) if n_done else None

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
        # ── S7 ATOMIC WRITE (crash-safe checkpoint) ──────────────────────────────
        # Write to a unique temp file in the SAME directory, flush+fsync to disk,
        # then os.replace() — an atomic rename on POSIX and Windows. A crash (or an
        # out-of-disk) mid-write leaves the temp file, never a half-written/corrupt
        # `path`, so the resume path always finds either the old good file or the
        # complete new one. Previously torch.save wrote `path` in place: a crash
        # during the write truncated the live checkpoint and the next resume loaded
        # a corrupt file (caught only after the fact by CheckpointManager.verify_all).
        import os as _os, tempfile as _tempfile
        path = str(path)
        d = _os.path.dirname(path) or "."
        _os.makedirs(d, exist_ok=True)
        fd, tmp = _tempfile.mkstemp(prefix=".ckpt_", suffix=".tmp", dir=d)
        _os.close(fd)
        try:
            with open(tmp, "wb") as f:
                torch.save(payload, f)
                f.flush()
                _os.fsync(f.fileno())
            _os.replace(tmp, path)        # atomic swap into place
        finally:
            if _os.path.exists(tmp):
                try:
                    _os.remove(tmp)
                except OSError:            # pragma: no cover
                    pass

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
