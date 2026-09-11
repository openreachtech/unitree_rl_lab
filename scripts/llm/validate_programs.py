"""Run skill programs on the ``Go2-Multitask-v2`` policy and measure what happened.

This is the simulator's half of the language-instruction dataset. Programs are compiled by
``unitree_rl_lab.program`` into the exact command stream the policy reads, each is executed on
``--replicas`` domain-randomised copies of the robot, and every replica is scored: did it stay on its
feet, did each flip land, did each stance come up, how far did each move actually carry. A program
whose replicas pass at ``--min-success`` or better is kept for the dataset.

The same run feeds the *capability table* -- per skill, the measured success rate and, for moves
and turns, a linear fit of distance against commanded duration. The compiler reads that fit back
so "5 m" becomes the number of seconds that really covers 5 m on this policy.

    # Measure every skill on its own and write the capability table
    python scripts/llm/validate_programs.py --calibrate --capability data/llm/capability.json

    # Validate sampled programs, using the table for distance -> time
    python scripts/llm/sample_programs.py --count 2000 --out data/llm/programs.jsonl
    python scripts/llm/validate_programs.py --programs data/llm/programs.jsonl \
        --capability data/llm/capability.json --out data/llm/validated.jsonl

Pass ``--checkpoint`` when the run's checkpoint is not under ``logs/rsl_rl/go2_multitask_v2``.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Go2-Multitask-v2")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--replicas", type=int, default=8, help="Domain-randomised copies each program runs on.")
parser.add_argument("--programs", type=str, default=None, help="JSONL of {id, program} records to validate.")
parser.add_argument("--calibrate", action="store_true",
                    help="Ignore --programs and run the canonical single-skill set that measures the capability table.")
parser.add_argument("--capability", type=str, default=None,
                    help="Capability table JSON. Read for the compiler's calibration when present; rewritten with\n"
                         "the measurements of this run.")
parser.add_argument("--out", type=str, default=None, help="Where to write per-program results (JSONL).")
parser.add_argument("--min-success", type=float, default=0.8,
                    help="Share of replicas that must pass for a program to be kept.")
parser.add_argument("--min-hold", type=float, default=0.5,
                    help="Share of a stance's hold the robot must spend up for the stance to count.")
parser.add_argument("--no-push", action="store_true",
                    help="Disable the periodic push disturbance. On by default for --calibrate, where a clean\n"
                         "speed measurement is the point; leave it on when validating, pushes happen for real.")
parser.add_argument("--limit", type=int, default=None, help="Only the first N programs of the file.")
parser.add_argument("--untrained", action="store_true",
                    help="Skip loading a checkpoint and run the freshly initialised network. The robot will fall;\n"
                         "this only exercises the command plumbing and scoring without a trained policy.")
AppLauncher.add_app_launcher_args(parser)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # noqa: E402

cli_args.add_rsl_rl_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402

# The simulator's shutdown can end the process before a block-buffered stdout is flushed, which
# loses the per-program lines when output is redirected to a file.
sys.stdout.reconfigure(line_buffering=True)
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.program import (  # noqa: E402
    DIRECTIONS,
    FLIP_KINDS,
    SPEEDS,
    STANCE_KINDS,
    CapabilityTable,
    CompilerConfig,
    Flip,
    Move,
    Stance,
    Timeline,
    Turn,
    compile_program,
    program_from_json,
    program_to_json,
)
from unitree_rl_lab.program.compiler import COL_FLIP, COL_STANCE, Segment  # noqa: E402
from unitree_rl_lab.program.grammar import FLIP_MOTION, STANCE_SIGN, step_to_dict  # noqa: E402
from unitree_rl_lab.tasks.dynamic.mdp.commands import JumpCommand  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


# =================================================================================================
# Programs
# =================================================================================================


def canonical_programs() -> list[dict]:
    """One program per skill and setting, at two lengths each so a slope and an offset can be fit."""
    out = []
    for direction in DIRECTIONS:
        for speed in SPEEDS:
            for seconds in (3.0, 6.0):
                out.append({"id": f"move:{direction}:{speed}:{seconds:g}s",
                            "program": [Move(dir=direction, speed=speed, duration_s=seconds)]})
    for direction in ("left", "right"):
        for speed in SPEEDS:
            for seconds in (2.0, 4.0):
                out.append({"id": f"turn:{direction}:{speed}:{seconds:g}s",
                            "program": [Turn(dir=direction, speed=speed, duration_s=seconds)]})
    for kind in FLIP_KINDS:
        for count in (1, 3):
            out.append({"id": f"flip:{kind}:x{count}", "program": [Flip(kind=kind, count=count)]})
    for kind in STANCE_KINDS:
        for seconds in (5.0, 10.0):
            out.append({"id": f"stance:{kind}:{seconds:g}s", "program": [Stance(kind=kind, duration_s=seconds)]})
    return out


def load_programs(path: str, limit: int | None) -> list[dict]:
    out = []
    with open(path) as handle:
        for line_number, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out.append({"id": record.get("id", f"line{line_number}"), "program": program_from_json(record["program"])})
            if limit is not None and len(out) >= limit:
                break
    return out


# =================================================================================================
# Environment
# =================================================================================================


def make_env(max_steps: int, dt_hint: float, disable_push: bool):
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    # The program is the only source of commands. Every self-scheduling path is switched off:
    # velocity resampling and heading control, the flip's own trigger schedule, the stance's own
    # trigger time. Their external setters do the rest.
    velocity_cfg = env_cfg.commands.base_velocity
    velocity_cfg.resampling_time_range = (1.0e9, 1.0e9)
    velocity_cfg.rel_standing_envs = 0.0
    velocity_cfg.heading_command = False
    env_cfg.commands.jump.auto_trigger = False
    env_cfg.commands.handstand.trigger_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.handstand.episode_probability = 1.0
    # Longer than any program in the batch, so the only resets are falls.
    env_cfg.episode_length_s = max_steps * dt_hint + 5.0
    if disable_push and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    # `bad_orientation` is gated on the stance's `enabled` flag with no settle after it, so it ends
    # the episode one step after every stance release, while the robot is still pitched 75 degrees
    # and has had no chance to come down (measured: every replica "fell" at release + 0.02 s). That
    # is a training-side artefact, not a fall. Here a fall is trunk contact, and whether the robot
    # is upright again is checked explicitly after each stance and at the end of the program.
    env_cfg.terminations.bad_orientation = None

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.untrained:
        resume_path = "<untrained>"
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if not args_cli.untrained:
        runner.load(resume_path)
    inference = runner.get_inference_policy(device=env.unwrapped.device)
    return env, inference, resume_path


# =================================================================================================
# Execution
# =================================================================================================


class BatchRun:
    """One batch: ``len(timelines) * replicas`` environments, stepped together for ``steps`` steps."""

    def __init__(self, env, inference, timelines: list[Timeline], replicas: int, steps: int):
        self.env = env
        self.inference = inference
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.dt = self.unwrapped.step_dt
        self.timelines = timelines
        self.replicas = replicas
        self.steps = steps
        self.num_envs = self.unwrapped.num_envs
        self.used = len(timelines) * replicas
        assert self.used <= self.num_envs

        self.velocity = self.unwrapped.command_manager.get_term("base_velocity")
        self.jump = self.unwrapped.command_manager.get_term("jump")
        self.handstand = self.unwrapped.command_manager.get_term("handstand")
        self.robot = self.unwrapped.scene["robot"]

        arrays = np.zeros((self.num_envs, steps, 5), dtype=np.float32)
        for p, timeline in enumerate(timelines):
            arrays[p * replicas:(p + 1) * replicas] = timeline.to_array(steps)
        self.cmd = torch.tensor(arrays, device=self.device)

        # Per program: the segments that get scored, with their step spans.
        self.scored: list[list[tuple[Segment, int, int]]] = []
        for timeline in timelines:
            spans = []
            for seg in timeline.segments:
                if seg.kind in ("move", "turn", "flip", "stance"):
                    start = int(round(seg.t0 / self.dt))
                    stop = min(max(int(round(seg.t1 / self.dt)), start + 1), steps)
                    spans.append((seg, start, stop))
            self.scored.append(spans)

    def env_ids(self, program: int) -> torch.Tensor:
        return torch.arange(program * self.replicas, (program + 1) * self.replicas, device=self.device)

    @torch.inference_mode()
    def run(self) -> list[dict]:
        # The whole batch, reset included, under one inference mode. Stepping under it and then
        # resetting outside it fails: the reset writes in place into buffers the step turned into
        # inference tensors ("Inplace update to inference tensor outside InferenceMode").
        env, dev, n = self.env, self.device, self.num_envs
        env.reset()
        obs = env.get_observations()
        self._flip_attempts_before = self.jump.total_attempts
        self._stance_attempts_before = self.handstand.attempts

        alive = torch.ones(n, dtype=torch.bool, device=dev)
        fell_step = torch.full((n,), -1, dtype=torch.long, device=dev)

        # Yaw integrated step by step, so a 360-degree turn does not wrap.
        yaw_prev = self._yaw()
        yaw_acc = torch.zeros(n, device=dev)

        # Flip attempts: an attempt opens on the trigger step and latches `success` until the
        # command re-arms (`trigger_step` drops back to -1) or the robot falls.
        max_flips = max([sum(1 for s, _, _ in spans if s.kind == "flip") for spans in self.scored] + [1])
        flip_ok = torch.zeros((n, max_flips), dtype=torch.bool, device=dev)
        flip_index = torch.full((n,), -1, dtype=torch.long, device=dev)
        flip_open = torch.zeros(n, dtype=torch.bool, device=dev)

        # Stances: steps spent in the stance, and whether it was ever reached.
        max_stances = max([sum(1 for s, _, _ in spans if s.kind == "stance") for spans in self.scored] + [1])
        stance_up_steps = torch.zeros((n, max_stances), device=dev)
        stance_reached = torch.zeros((n, max_stances), dtype=torch.bool, device=dev)
        stance_index = torch.full((n,), -1, dtype=torch.long, device=dev)
        # Steps from a stance's release until the trunk is upright again; -1 while still coming down.
        stance_recover = torch.full((n, max_stances), -1, dtype=torch.long, device=dev)
        release_step = torch.full((n,), -1, dtype=torch.long, device=dev)

        # Motion segments: pose at the start, displacement at the end.
        start_pose: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        motion_result: dict[tuple[int, int], torch.Tensor] = {}  # (program, seg idx) -> (replicas, 3)

        # Fast lookup of segment starts/ends per step.
        starts: dict[int, list[tuple[int, int, Segment]]] = {}
        ends: dict[int, list[tuple[int, int, Segment]]] = {}
        for p, spans in enumerate(self.scored):
            for k, (seg, start, stop) in enumerate(spans):
                starts.setdefault(start, []).append((p, k, seg))
                ends.setdefault(stop - 1, []).append((p, k, seg))

        flip_codes = {kind: FLIP_MOTION[kind] for kind in FLIP_KINDS}
        target_height_jump = float(self.jump.cfg.target_height_range[0])
        flip_height = float(self.jump.cfg.flip_target_height)
        flip_launch = float(self.jump.cfg.flip_launch_height) if self.jump.cfg.flip_launch_height > 0 else flip_height

        for t in range(self.steps):
            # -- commands for this step, before the physics that will see them -------------------
            self.velocity.vel_command_b[:] = self.cmd[:, t, :3]

            for p, k, seg in starts.get(t, []):
                ids = self.env_ids(p)
                if seg.kind in ("move", "turn"):
                    start_pose[(p, k)] = (self.robot.data.root_pos_w[ids, :2].clone(), yaw_acc[ids].clone(), self._yaw()[ids].clone())
                elif seg.kind == "flip":
                    code, pitch, roll = flip_codes[seg.flip]
                    self.jump.motion_code[ids] = code
                    height = target_height_jump if seg.flip == "jump" else flip_height
                    self.jump.launch_height[ids] = target_height_jump if seg.flip == "jump" else flip_launch
                    self.jump.set_command(ids, True, target_height=height, target_pitch_turns=pitch, target_roll_turns=roll)
                    flip_index[ids] += 1
                    flip_open[ids] = True
                elif seg.kind == "stance":
                    self.handstand.set_command(ids, True, stance=seg.stance, hold_duration=seg.duration)
                    stance_index[ids] += 1

            # -- step ----------------------------------------------------------------------------
            obs, _, dones, _ = env.step(self.inference(obs))

            fell = dones.bool() & alive
            fell_step[fell] = t
            alive &= ~fell

            yaw_now = self._yaw()
            yaw_acc += wrap_to_pi(yaw_now - yaw_prev) * alive.float()
            yaw_prev = yaw_now

            # -- score -------------------------------------------------------------------------
            opened = flip_open & (flip_index >= 0)
            if torch.any(opened):
                rows = opened.nonzero(as_tuple=False).flatten()
                flip_ok[rows, flip_index[rows]] |= self.jump.success[rows] & alive[rows]
                closed = opened & ((self.jump.trigger_step < 0) | ~alive)
                flip_open &= ~closed

            # `alive` matters: a fallen robot is reset standing, which is not a recovery.
            recovering = (release_step >= 0) & (stance_index >= 0) & alive
            if torch.any(recovering):
                rows = recovering.nonzero(as_tuple=False).flatten()
                done_rows = rows[self._upright()[rows] & (stance_recover[rows, stance_index[rows]] < 0)]
                stance_recover[done_rows, stance_index[done_rows]] = t - release_step[done_rows]
                release_step[done_rows] = -1

            in_stance = (self.cmd[:, t, COL_STANCE] != 0) & (stance_index >= 0)
            if torch.any(in_stance):
                rows = in_stance.nonzero(as_tuple=False).flatten()
                up = self.handstand.success[rows] & alive[rows]
                stance_up_steps[rows, stance_index[rows]] += up.float()
                stance_reached[rows, stance_index[rows]] |= up

            for p, k, seg in ends.get(t, []):
                ids = self.env_ids(p)
                if seg.kind in ("move", "turn"):
                    pos0, yaw_acc0, yaw0 = start_pose.pop((p, k))
                    delta_w = self.robot.data.root_pos_w[ids, :2] - pos0
                    c, s = torch.cos(yaw0), torch.sin(yaw0)
                    dx = c * delta_w[:, 0] + s * delta_w[:, 1]
                    dy = -s * delta_w[:, 0] + c * delta_w[:, 1]
                    dyaw = yaw_acc[ids] - yaw_acc0
                    motion_result[(p, k)] = torch.stack((dx, dy, dyaw), dim=-1)
                elif seg.kind == "stance":
                    self.handstand.set_command(ids, False)
                    release_step[ids] = t

        # The terms keep their own tallies (scored at re-arm / at the end of a hold). If they
        # disagree with what was fired, the triggers are not reaching the policy and every rate
        # above is meaningless -- so say so loudly rather than report zeros.
        fired_flips = int((self.cmd[:self.used, :, COL_FLIP] != 0).sum().item())
        fired_stances = sum(sum(1 for s, _, _ in spans if s.kind == "stance") for spans in self.scored) * self.replicas
        seen_flips = self.jump.total_attempts - self._flip_attempts_before
        seen_stances = self.handstand.attempts - self._stance_attempts_before
        print(f"  [check] flips fired {fired_flips}, registered by the jump command {seen_flips};"
              f" stances fired {fired_stances}, registered by the handstand command {seen_stances}")
        if seen_flips < fired_flips * 0.9 or seen_stances < fired_stances * 0.9:
            print("  [check] WARNING: the command terms did not see every trigger -- results are not trustworthy")

        ended_upright = alive & self._upright()
        return self._collect(alive, fell_step, flip_ok, stance_up_steps, stance_reached, stance_recover,
                             motion_result, ended_upright)

    def _yaw(self) -> torch.Tensor:
        return euler_xyz_from_quat(self.robot.data.root_quat_w)[2]

    def _upright(self) -> torch.Tensor:
        """Trunk within about 25 degrees of level: gravity's body-z component below -0.9."""
        return self.robot.data.projected_gravity_b[:, 2] < -0.9

    def _collect(self, alive, fell_step, flip_ok, stance_up_steps, stance_reached, stance_recover, motion_result,
                 ended_upright) -> list[dict]:
        results = []
        for p, (timeline, spans) in enumerate(zip(self.timelines, self.scored)):
            ids = self.env_ids(p)
            survived = alive[ids]
            fell_at = fell_step[ids]
            upright_end = ended_upright[ids]
            passed = survived & upright_end
            flips, stances, motions = [], [], []
            flip_k = stance_k = 0
            for k, (seg, start, stop) in enumerate(spans):
                fell_during = (fell_at >= start) & (fell_at < stop)
                if seg.kind == "flip":
                    ok = flip_ok[ids, flip_k]
                    passed &= ok
                    flips.append({"kind": seg.flip, "t": round(seg.t0, 2), "success_rate": ok.float().mean().item(),
                                  "fall_rate": fell_during.float().mean().item(), "ok": ok.tolist(), "fell": fell_during.tolist()})
                    flip_k += 1
                elif seg.kind == "stance":
                    hold = stance_up_steps[ids, stance_k] / max(stop - start, 1)
                    reached = stance_reached[ids, stance_k]
                    recover = stance_recover[ids, stance_k]
                    ok = reached & (hold >= args_cli.min_hold)
                    passed &= ok
                    stances.append({"kind": seg.label, "t": round(seg.t0, 2), "duration_s": round(seg.duration, 2),
                                    "success_rate": ok.float().mean().item(), "reached_rate": reached.float().mean().item(),
                                    "hold_fraction": hold.tolist(), "fall_rate": fell_during.float().mean().item(),
                                    "recover_s": [round(v * self.dt, 2) if v >= 0 else None for v in recover.tolist()],
                                    "ok": ok.tolist(), "fell": fell_during.tolist()})
                    stance_k += 1
                else:
                    measured = motion_result[(p, k)]
                    valid = (fell_at < 0) | (fell_at >= stop)
                    direction, speed = seg.label.split()
                    if seg.kind == "move":
                        norm = math.hypot(seg.vx, seg.vy)
                        along = (measured[:, 0] * seg.vx + measured[:, 1] * seg.vy) / norm
                        rate = norm
                    else:
                        along = measured[:, 2] * (1.0 if seg.wz > 0 else -1.0)
                        rate = abs(seg.wz)
                    motions.append({"kind": seg.kind, "dir": direction, "speed": speed, "t": round(seg.t0, 2),
                                    "duration_s": round(seg.duration, 3), "commanded_rate": rate,
                                    "commanded": [round(v, 3) for v in seg.commanded_delta()],
                                    "measured_mean": measured[valid].mean(dim=0).tolist() if valid.any() else None,
                                    "along": along.tolist(), "valid": valid.tolist(),
                                    "fall_rate": fell_during.float().mean().item()})
            results.append({
                "survival_rate": survived.float().mean().item(),
                "ended_upright_rate": upright_end.float().mean().item(),
                "ended_upright": upright_end.tolist(),
                "replica_passed": passed.tolist(),
                "success_rate": passed.float().mean().item(),
                "passed": passed.float().mean().item() >= args_cli.min_success,
                "fell_at_s": [round(v * self.dt, 2) if v >= 0 else None for v in fell_at.tolist()],
                "flips": flips,
                "stances": stances,
                "motions": motions,
            })
        return results


# =================================================================================================
# Main
# =================================================================================================


def record_capability(table: CapabilityTable, result: dict) -> None:
    for flip in result["flips"]:
        for ok, fell in zip(flip["ok"], flip["fell"]):
            table.record_event(f"flip:{flip['kind']}", success=ok, fell=fell)
    for stance in result["stances"]:
        for ok, fell, hold, recover in zip(stance["ok"], stance["fell"], stance["hold_fraction"], stance["recover_s"]):
            table.record_event(f"stance:{stance['kind']}", success=ok, fell=fell, hold_fraction=hold, recover_s=recover)
    for motion in result["motions"]:
        key = f"{motion['kind']}:{motion['dir']}:{motion['speed']}"
        for along, valid in zip(motion["along"], motion["valid"]):
            table.record_motion(key, motion["commanded_rate"], motion["duration_s"], along, fell=not valid)


def main():
    for kind, (code, _, _) in FLIP_MOTION.items():
        expected = {"jump": JumpCommand.MOTION_JUMP, "backflip": JumpCommand.MOTION_BACKFLIP,
                    "sideflip_left": JumpCommand.MOTION_SIDEFLIP, "frontflip": JumpCommand.MOTION_HANDSPRING,
                    "sideflip_right": JumpCommand.MOTION_SIDEFLIP_RIGHT}[kind]
        assert code == expected, f"FLIP_MOTION[{kind}] = {code} but JumpCommand says {expected}"

    if args_cli.calibrate:
        records = canonical_programs()
        disable_push = True
    else:
        if not args_cli.programs:
            raise SystemExit("--programs or --calibrate is required")
        records = load_programs(args_cli.programs, args_cli.limit)
        disable_push = args_cli.no_push

    table = CapabilityTable.load_or_empty(args_cli.capability)
    compiler = CompilerConfig(calibration=table.calibration())
    if args_cli.calibrate:
        # Measuring the table: compile against nominal speeds, not against the previous table.
        compiler = CompilerConfig()
        table = CapabilityTable()

    timelines = []
    for record in records:
        record["timeline"] = compile_program(record["program"], compiler)
    steps = max(record["timeline"].num_steps for record in records)
    print(f"[INFO] {len(records)} programs, longest {steps * compiler.dt:.1f} s, {args_cli.replicas} replicas each")

    env, inference, checkpoint = make_env(steps, compiler.dt, disable_push)
    per_batch = env.unwrapped.num_envs // args_cli.replicas
    if per_batch == 0:
        raise SystemExit(f"--num_envs {args_cli.num_envs} is smaller than --replicas {args_cli.replicas}")

    out_handle = open(args_cli.out, "w") if args_cli.out else None
    kept = 0
    for batch_start in range(0, len(records), per_batch):
        batch = records[batch_start:batch_start + per_batch]
        run = BatchRun(env, inference, [r["timeline"] for r in batch], args_cli.replicas, steps)
        results = run.run()
        for record, result in zip(batch, results):
            record_capability(table, result)
            kept += int(result["passed"])
            flips = " ".join(f"{f['kind']}={f['success_rate']:.2f}" for f in result["flips"])
            stances = " ".join(f"{s['kind']}={s['success_rate']:.2f}" for s in result["stances"])
            print(f"  {record['id']:<40} survive {result['survival_rate']:.2f}  upright {result['ended_upright_rate']:.2f}"
                  f"  pass {result['success_rate']:.2f}"
                  f"  {'KEEP' if result['passed'] else 'drop'}  {flips} {stances}")
            if out_handle:
                out_handle.write(json.dumps({
                    "id": record["id"],
                    "program": [step_to_dict(s) for s in record["program"]],
                    "timeline": record["timeline"].to_dict(),
                    "replicas": args_cli.replicas,
                    **{k: v for k, v in result.items()},
                }, ensure_ascii=False) + "\n")
        print(f"[INFO] batch {batch_start // per_batch + 1}: {kept}/{batch_start + len(batch)} kept so far")

    if out_handle:
        out_handle.close()
    env.close()

    table.meta.update({"task": args_cli.task, "checkpoint": str(checkpoint), "replicas": args_cli.replicas,
                       "mode": "calibrate" if args_cli.calibrate else "validate", "push_disabled": disable_push})
    print()
    print(table.summary())
    if args_cli.capability:
        Path(args_cli.capability).parent.mkdir(parents=True, exist_ok=True)
        table.save(args_cli.capability)
        print(f"[INFO] capability table written to {args_cli.capability}")
    print(f"[INFO] {kept}/{len(records)} programs passed at >= {args_cli.min_success:.0%}")


if __name__ == "__main__":
    main()
    simulation_app.close()
