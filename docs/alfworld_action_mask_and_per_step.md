# Alfworld Action Mask and Per-Step Rewards

## Action-Mask Rollouts (Implemented)

For multi-turn environments like Alfworld, actions must be conditioned on the latest observation. To preserve that
conditioning while avoiding training on observation tokens, the rollout returns a single full-episode sequence in
`completion_ids` and an `action_mask` that marks which tokens are actions.

### Rollout Expectations

- `prompt_ids`: the initial prompt tokens for the first turn.
- `completion_ids`: a full episode sequence that interleaves action tokens and observation tokens:
  `a1, o1, a2, o2, ..., aN`.
- `action_mask`: same length as `completion_ids`.
  - `1` for action tokens.
  - `0` for observation or chat-template tokens.
- `logprobs`: same length as `completion_ids`.
  - Use real logprobs for action tokens.
  - Use `0.0` for non-action tokens.

This lets the model attend to observations (they remain in the input sequence), while the trainer excludes them from
loss, importance sampling, and metrics.

### Trainer Expectations

The custom Axolotl trainer reads `action_mask` and:

- Keeps `completion_mask` unchanged for attention.
- Computes a loss mask: `completion_mask * action_mask`.
- Uses the loss mask for importance sampling, loss, and metrics.

This is implemented in `dockerfiles/patches/axolotl_grpo_rollout_fix.py`.

## Per-Step Rewards (Not Implemented Yet)

If you want to train with step-wise rewards instead of a single episode reward, you need per-token (or per-action)
advantages instead of a single scalar per sequence.

Two common approaches:

1) **Repeat per-step rewards over action tokens**
   - Compute a reward per action (step reward).
   - Expand each action’s reward across its action tokens.
   - Build a token-level advantage vector aligned to `completion_ids`.

2) **Compute advantages directly in the trainer**
   - Return `step_rewards` (per action) in the rollout’s extra fields.
   - Modify the trainer’s `_generate_and_score_completions` to:
     - Align step rewards with action tokens via `action_mask`.
     - Produce `advantages` with shape `(B, T)` instead of `(B,)`.
   - GRPO already supports `(B, T)` advantages in `_compute_loss`.

Practical notes:

- You still need the `action_mask` to align rewards to action tokens.
- If you keep reward normalization, apply it at the action/step level before token expansion.
- If using vLLM importance sampling, ensure masked tokens do not contribute to IS ratios.

Once you decide the exact per-step reward definition, the safest path is to extend the action-mask trainer to compute
token-level advantages and pass them into loss as `(B, T)`.
