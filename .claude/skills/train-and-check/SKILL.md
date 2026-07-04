---
name: train-and-check
description: Run RL training for a Unitree task and print aggregated TensorBoard results. Use when the user asks to train a task and check results, or says "train and aggregate", "train and check", or mentions train_and_aggregate.py.
---

# Train and Check Results

Runs `scripts/rsl_rl/train_and_aggregate.py` and prints the aggregated metric summary.
Console output from training and the aggregator is suppressed; only the final summary is printed.

## Arguments

Pass flags directly to `train_and_aggregate.py`:

| Flag | Required | Default | Description |
|---|---|---|---|
| `--task` | Yes | — | Gym task ID (e.g. `Unitree-Go2-Velocity-v1-Phase2`) |
| `--previous-task` | No | — | Resume from this task's latest checkpoint |
| `--max_iterations` | No | 3000 | Number of training iterations |
| `--aggregate-interval` | No | 100 | Log aggregation interval |

Fixed internally: `--num_envs 4096 --headless`.

## Steps

1. Identify `--task` from the user's message.
2. If the user mentions a previous phase or checkpoint source, add `--previous-task`.
3. If `--max_iterations` was specified, pass it; otherwise omit (default 3000).
4. Run from the workspace root:

```bash
python scripts/rsl_rl/train_and_aggregate.py --task <TASK> [--previous-task <PREV>] [--max_iterations <N>]
```

5. Print the command output to the user as-is.

If `--task` is missing, ask the user before running.

## Example

```bash
python scripts/rsl_rl/train_and_aggregate.py \
  --task Unitree-Go2-Velocity-v1-Phase2 \
  --previous-task Unitree-Go2-Velocity-v1-Phase1 \
  --max_iterations 5000
```
