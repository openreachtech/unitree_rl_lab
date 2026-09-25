#!/usr/bin/env python3
"""run の Policy/mean_noise_std と action_rate を iter ごとに吐く（収穫帯の起点選び用）。"""
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]
g = lambda t: {s.step: s.value for s in ea.Scalars(t)} if t in tags else {}
std, ar = g("Policy/mean_noise_std"), g("Episode_Reward/action_rate")
for i in sorted(std):
    print(f"{i}\t{std[i]:.4f}\t{ar.get(i, float('nan')):.3f}")
