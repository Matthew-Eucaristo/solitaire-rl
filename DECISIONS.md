# Decisions log

Defaults chosen where the brief left a choice open. Each entry notes the
alternative and why the default was picked.

| # | Decision | Chosen | Rationale |
|---|----------|--------|-----------|
| D1 | Stock order in the episode-signature | Included (internal only) | Cycle detection must see the *full* physical state. Two boards with the same visible cards but different stock order are not equivalent states. The signature is used only for termination — it never enters the observation (POMDP correctness is tested). |
| D2 | `redeals` in the signature | Excluded | A repeat of the exact board IS a cycle regardless of how many redeals it took; including the count made detection unreachable (each redeal produced a fresh signature). |
| D3 | `CONCEDE` always legal | Yes, id 653 | The give-up behavior Matthew asked for; an agent must be able to resign even when moves exist. |
| D4 | Random baseline may pick CONCEDE | No | Uniform-with-concede resigns ~1/654 of the time from every state and ~always dies on cycle first anyway; the interesting random baseline is "random player that doesn't quit on purpose". Documented in `RandomPolicy`. |
| D5 | Headline variant | draw-1 | Brief's open question, default given: headline draw-1, draw-3 implemented too. |
| D6 | Redeal order | `stock = waste[::-1]` | Preserves original draw sequence — the common convention. |
| D7 | Partial run moves | Allowed | Any face-up card + its run may move (standard Klondike). |
| D8 | MAX_UP cap | 13 | A face-up run legally can't exceed 13 distinct alternating ranks... in practice 7+6=13 is the column-size bound needed by the T2T index scheme (7×7×13 ids). |
| D9 | Frame stack | Inside the env (`frame_stack=8`, `_obs()` = last 8 raw frames concatenated) | Simplest reliable route for both DQN and MaskablePPO (vec envs forward it transparently). k is configurable; raw obs accessible for policy-free code. |
| D10 | POMDP obs layout | 6163-dim: one-hot tableau (7×13×53) + down counts (7) + waste order (24×53) + foundations (4×14) + stock count + scalars | Cards as one-hot over position (52+unknown); stock order omitted by design — tested. Perfect adds 3498 dims of stock order + face-down contents. |
| D11 | Train seeds | `seed ≥ 1_000_000`, sampled by each env instance | Guarantees train ∩ benchmark = ∅ by construction (benchmark = seeds 0–999). |
| D12 | Eval parallelism | `multiprocessing` spawn pool | Policies are constructed per-worker; checkpoint policies load their net in the worker. |
| D13 | DQN | Hand-rolled Double-DQN, masked target + masked ε-greedy, 2×256 MLP, uint8 replay storage | Keeps dependencies at the brief's ceiling; masking is identical at sample/target time so the argmax never selects masked actions. |
| D14 | Replay buffer obs storage | `uint8` (obs scaled ×255) | 49k-dim obs → float32 replay of 30k transitions would be ~5.9 GB; uint8 is 1.5 GB, fits the 16 GB laptop target. |
| D15 | `no_progress_idle` budget | 150 moves | Generous enough not to kill winning trajectories (~230 avg moves/heuristic win); tight enough to end hopeless loops. |
| D16 | CI | ubuntu-latest only, `uv sync --extra dev --frozen`, ruff + pytest | Brief §0; runtime well under 2 min. |
| D17 | Dependencies beyond the brief | `pillow`, `matplotlib`, `tensorboard` | GIF recordings (user requirement: ≥5 solve videos), learning-curve figures, PPO's TB logging. All light; nothing heavier added. |
| D18 | Web app stack | FastAPI + uvicorn serving the Python engine itself; vanilla JS/CSS UI in `web/` | Zero rules duplication — no JS engine port that could diverge from the tested engine. Agent steps rebuild the policy's env by replaying the game's action history with the policy's own obs variant, so watch-mode is identical to eval. Deps added beyond the brief: `fastapi`, `uvicorn`, `httpx` (dev). |
