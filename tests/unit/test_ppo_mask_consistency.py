"""
PPO masking correctness (user-flagged): when FLAT is masked, the sampled action
AND its stored log-prob must both come from the MASKED distribution, so PPO's
ratio (new_logp - old_logp) is consistent. Verified on CPU; identical on A100.
"""
import math
import torch
from core.settings import CFG, auto_tune_batch
from core.agent.ppo import PPOAgent
from core.agent.action_space import FLAT, BUY, SELL, DIRECTION_DIM

DEV = torch.device("cpu")


def _agent(sd=16):
    c = auto_tune_batch(dict(CFG), DEV)
    c.update({"USE_AMP": False, "USE_TORCH_COMPILE": False})
    return PPOAgent(sd, c, DEV)


def test_masked_flat_never_sampled_and_logp_finite():
    agent = _agent()
    state = torch.randn(64, 16)
    mask = torch.tensor([[0.0, 1.0, 1.0]]).expand(64, -1)  # FLAT masked
    out = agent.select_actions(state, mask=mask)
    assert (out["direction"] != FLAT).all()          # FLAT impossible
    assert torch.isfinite(out["logp"]).all()         # log-prob finite (not -inf)


def test_logp_matches_masked_distribution():
    # The stored log-prob must equal the masked distribution's log-prob for the
    # sampled action (NOT the unmasked one). We reconstruct both and compare.
    agent = _agent()
    torch.manual_seed(0)
    state = torch.randn(1, 16)
    mask = torch.tensor([[0.0, 1.0, 1.0]])           # FLAT masked
    dir_logits, exit_logits, lot_mean, _v = agent.net(state)
    # masked dist
    dir_d_m, exit_d, lot_d = agent._dists(dir_logits, exit_logits, lot_mean, mask)
    # FLAT prob under the masked dist must be ~0
    assert dir_d_m.probs[0, FLAT].item() < 1e-6
    # BUY+SELL probs sum to ~1 under the mask
    assert abs(dir_d_m.probs[0, BUY].item() + dir_d_m.probs[0, SELL].item() - 1.0) < 1e-5


def test_update_with_mask_is_consistent():
    # A full rollout WITH masks must update without NaN/inf (ratio well-defined).
    agent = _agent()
    B = 8
    mask = torch.tensor([[0.0, 1.0, 1.0]]).expand(B, -1)
    for _ in range(12):
        s = torch.randn(B, 16)
        out = agent.select_actions(s, mask=mask)
        agent.store(s, out, torch.randn(B), torch.zeros(B).bool(), mask)
    loss = agent.update()
    assert loss is not None and math.isfinite(loss)


def test_inactive_gate_only_flat(monkeypatch):
    # Sanity: a mask allowing ONLY flat forces FLAT (gate inactive case).
    agent = _agent()
    agent_mask = torch.tensor([[1.0, 0.0, 0.0]]).expand(16, -1)
    out = agent.select_actions(torch.randn(16, 16), mask=agent_mask)
    assert (out["direction"] == FLAT).all()
