"""Foothold-gated height patches (Heyrman et al., "Watch Your Step", 2026).

``FootholdPatches`` is one observation term that does the whole perception pipeline:

1. **Map.** An ``OdomElevationMap`` (the MID-360 fan with feat/mid360's sensor and odometry
   noise, binned into an odom-aligned grid that scrolls with the robot) large enough to keep
   what the sensor saw beyond its blind disc until the robot walks onto it. Read as
   ``root_z - terrain_z - offset`` (0 on flat ground at nominal stance, negative on a step up).
2. **Footholds.** The foothold prediction model (FPM), an MLP from noise-free
   proprioception and the command to the next two footholds ``(x, y)`` in the yaw-aligned
   base frame. Blind by design: it sees neither the map nor its own previous output.
3. **Patches.** ``k x k`` points on the 5 cm yaw-aligned base-frame lattice around each
   predicted foothold (snapped to the lattice), each read from the nearest map cell. Output ``[p1 (2), patch1 (k*k), p2 (2), patch2 (k*k)]``.
4. **Co-training.** Every step records the FPM input, foot touchdowns and poses into a ring
   buffer. The footholds are pinned under the base until the policy walks: at least
   ``bootstrap_iterations`` have passed and, over the last ``walk_check_iterations``, at least
   ``walk_threshold`` of the finished episodes ran to the time limit. Then (and every
   ``refit_interval`` iterations after) the FPM is refit from scratch on the buffer (labels: the next
   two touchdowns strictly after each state, of either foot; windows that cross a reset are
   dropped) and swapped in.

The FPM is saved next to the policy checkpoints as ``fpm_<iteration>.pt`` when
``env.fpm_save_dir`` is set (``scripts/rsl_rl/train.py``), and loaded from
``env.fpm_load_dir`` (resume / play): the newest file at or below ``env.fpm_load_iteration``.
"""

from __future__ import annotations

import glob
import os
import re
import torch
import torch.nn as nn
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from .lidar_elevation_map import OdomElevationMap, _yaw_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["FootholdPatches", "FootholdModel", "fpm_val_error_1", "fpm_val_error_2", "fpm_label_rate", "fpm_walk_rate"]


class FootholdModel(nn.Module):
    """MLP [128, 128], LayerNorm + LeakyReLU(0.1), orthogonal init; z-scores its input with
    statistics stored in the module, so the exported model needs no external normalisation."""

    def __init__(self, num_inputs: int, num_outputs: int = 4, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        d = num_inputs
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
            d = h
        layers.append(nn.Linear(d, num_outputs))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
        self.register_buffer("mean", torch.zeros(num_inputs))
        self.register_buffer("std", torch.ones(num_inputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


class FootholdPatches(ManagerTermBase):
    """See the module docstring. One instance per env; the same group feeds actor and critic."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._asset_cfg: SceneEntityCfg = p["asset_cfg"]
        self._feet_cfg: SceneEntityCfg = p["feet_cfg"]
        self._contact_cfg: SceneEntityCfg = p["contact_cfg"]
        self._asset_cfg.resolve(env.scene)
        self._feet_cfg.resolve(env.scene)
        self._contact_cfg.resolve(env.scene)
        self._command_name: str = p["command_name"]
        self._k: int = p["patch_size"]
        self._offset: float = p["map_params"]["offset"]

        # -- map, driven from here
        self._map = OdomElevationMap(env, feet_cfg=self._feet_cfg, asset_cfg=self._asset_cfg, **p["map_params"])
        self._res: float = p["map_params"]["resolution"]

        # -- FPM
        self._num_inputs = 3 + 3 + 3 + 2 * env.scene[self._asset_cfg.name].num_joints
        self._fpm = FootholdModel(self._num_inputs).to(self.device)
        self._fpm_ready = False
        self._bootstrap = torch.tensor(p["bootstrap_footholds"], device=self.device).flatten()  # (4,)
        self._steps_per_iter: int = p["num_steps_per_env"]
        self._bootstrap_iters: int = p["bootstrap_iterations"]
        self._refit_iters: int = p["refit_interval"]
        self._train: bool = p["train"]
        self._fit = dict(p["fit"])

        # -- rollout buffer (ring, ``buffer_steps`` long)
        b, n, j = p["buffer_steps"], self.num_envs, self._num_inputs
        self._B = b
        self._buf_x = torch.zeros(b, n, j, device=self.device)
        self._buf_touch = torch.zeros(b, n, 2, dtype=torch.bool, device=self.device)
        self._buf_foot_xy = torch.zeros(b, n, 2, 2, device=self.device)
        self._buf_root_xy = torch.zeros(b, n, 2, device=self.device)
        self._buf_yaw = torch.zeros(b, n, device=self.device)
        self._buf_new_ep = torch.zeros(b, n, dtype=torch.bool, device=self.device)
        self._buf_pos = 0
        self._buf_count = 0
        self._min_air: float = p["min_air_time"]
        self._next_refit_step = self._bootstrap_iters * self._steps_per_iter
        # -- "can it walk yet": time-out share of the episodes finished in a window
        self._walk_threshold: float = p["walk_threshold"]
        self._walk_check_steps: int = p["walk_check_iterations"] * self._steps_per_iter
        self._walking = False
        self._ends = 0
        self._timeouts = 0
        self._next_walk_check = self._walk_check_steps
        env.fpm_walk_rate = 0.0
        self._last_counter = -1

        # patch cell offsets (k x k, x outer, y inner), centred
        r = torch.arange(self._k, device=self.device) - self._k // 2
        self._dx, self._dy = torch.meshgrid(r, r, indexing="ij")
        self._dx, self._dy = self._dx.flatten(), self._dy.flatten()

        env.fpm_val_error = (0.0, 0.0)
        env.fpm_label_rate = 0.0
        self._load_checked = False  # the scripts set env.fpm_load_dir after the env is built

    # ------------------------------------------------------------------ io
    def _try_load(self, env: ManagerBasedRLEnv) -> None:
        d = getattr(env, "fpm_load_dir", None)
        if not d:
            return
        limit = getattr(env, "fpm_load_iteration", None)
        best, best_it = None, -1
        for f in glob.glob(os.path.join(d, "fpm_*.pt")):
            m = re.match(r"fpm_(\d+)\.pt$", os.path.basename(f))
            if m and int(m.group(1)) > best_it and (limit is None or int(m.group(1)) <= limit):
                best, best_it = f, int(m.group(1))
        if best is None:
            print(f"[FPM] no fpm_*.pt in {d}; footholds stay pinned until the first refit")
            return
        self._fpm.load_state_dict(torch.load(best, map_location=self.device))
        self._fpm_ready = True
        self._walking = True
        # resumed run: the counter restarts at 0, refit after one interval
        self._next_refit_step = self._refit_iters * self._steps_per_iter
        print(f"[FPM] loaded {best}")

    def _save(self, env: ManagerBasedRLEnv, iteration: int) -> None:
        d = getattr(env, "fpm_save_dir", None)
        if not d:
            return
        os.makedirs(d, exist_ok=True)
        torch.save(self._fpm.state_dict(), os.path.join(d, f"fpm_{iteration}.pt"))

    # ------------------------------------------------------------------ helpers
    def _fpm_input(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Noise-free, unscaled: ang vel (base), projected gravity, command, joint pos / vel
        relative to the default pose."""
        a = env.scene[self._asset_cfg.name]
        return torch.cat(
            [
                a.data.root_ang_vel_b,
                a.data.projected_gravity_b,
                env.command_manager.get_command(self._command_name)[:, :3],
                a.data.joint_pos - a.data.default_joint_pos,
                a.data.joint_vel - a.data.default_joint_vel,
            ],
            dim=-1,
        )

    def _patch_points(self, footholds: torch.Tensor) -> torch.Tensor:
        """(N, 4) footholds -> (N, 2, k*k, 2) lattice points in the yaw-aligned base frame."""
        p = footholds.view(-1, 2, 1, 2)
        snapped = torch.round(p / self._res)
        offsets = torch.stack([self._dx, self._dy], dim=-1).view(1, 1, -1, 2)
        return (snapped + offsets) * self._res

    def _sample(self, footholds: torch.Tensor) -> torch.Tensor:
        """``footholds`` (N, 4) -> (N, 2 * (2 + k*k))."""
        n, kk = footholds.shape[0], self._k * self._k
        h = self._map.sample(self._patch_points(footholds).view(n, 2 * kk, 2)).view(n, 2, kk)
        return torch.cat([footholds[:, :2], h[:, 0], footholds[:, 2:], h[:, 1]], dim=-1)

    def _record(self, env: ManagerBasedRLEnv, x: torch.Tensor) -> None:
        a = env.scene[self._asset_cfg.name]
        contact = env.scene.sensors[self._contact_cfg.name]
        first = contact.compute_first_contact(env.step_dt)[:, self._contact_cfg.body_ids]
        air = contact.data.last_air_time[:, self._contact_cfg.body_ids]
        i = self._buf_pos
        self._buf_x[i] = x
        self._buf_touch[i] = first & (air > self._min_air)
        self._buf_foot_xy[i] = a.data.body_pos_w[:, self._feet_cfg.body_ids, :2]
        self._buf_root_xy[i] = a.data.root_pos_w[:, :2]
        self._buf_yaw[i] = _yaw_from_quat(a.data.root_quat_w)
        self._buf_new_ep[i] = env.episode_length_buf == 0
        self._buf_pos = (i + 1) % self._B
        self._buf_count = min(self._buf_count + 1, self._B)

    # ------------------------------------------------------------------ refit
    @torch.no_grad()
    def _labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Dataset from the ring buffer: (x, p) with p the next two touchdowns after each state."""
        order = (torch.arange(self._B, device=self.device) + self._buf_pos) % self._B  # oldest first
        x = self._buf_x[order]  # (B, N, J)
        touch = self._buf_touch[order]  # (B, N, 2)
        foot_xy = self._buf_foot_xy[order]
        root_xy, yaw = self._buf_root_xy[order], self._buf_yaw[order]
        episode = torch.cumsum(self._buf_new_ep[order].long(), dim=0)  # (B, N)
        B, N = touch.shape[:2]

        # touchdown events per env, in time order: (N, E) times and xy, padded with time = inf
        t_idx, n_idx, f_idx = touch.nonzero(as_tuple=True)
        key = n_idx * (2 * B) + t_idx * 2 + f_idx  # sort by env, then time
        key, perm = torch.sort(key)
        t_idx, n_idx, f_idx = t_idx[perm], n_idx[perm], f_idx[perm]
        counts = torch.bincount(n_idx, minlength=N)
        E = int(counts.max()) if counts.numel() else 0
        if E < 2:
            return x.new_zeros(0, x.shape[-1]), x.new_zeros(0, 4)
        start = torch.cumsum(counts, 0) - counts
        slot = torch.arange(t_idx.numel(), device=self.device) - start[n_idx]
        ev_t = torch.full((N, E), float(B + 1), device=self.device)
        ev_t[n_idx, slot] = t_idx.float()
        ev_xy = torch.zeros(N, E, 2, device=self.device)
        ev_xy[n_idx, slot] = foot_xy[t_idx, n_idx, f_idx]
        ev_ep = torch.full((N, E), -1, dtype=torch.long, device=self.device)
        ev_ep[n_idx, slot] = episode[t_idx, n_idx]

        # for each state t, the first event strictly after t
        q = torch.arange(B, device=self.device, dtype=torch.float).expand(N, B).contiguous()
        first = torch.searchsorted(ev_t.contiguous(), q, right=True)  # (N, B)
        i1, i2 = first.clamp(max=E - 1), (first + 1).clamp(max=E - 1)
        ok = (first + 1 < E) & (torch.gather(ev_t, 1, i2) <= B - 1)
        ep_now = episode.t()  # (N, B)
        ok &= (torch.gather(ev_ep, 1, i1) == ep_now) & (torch.gather(ev_ep, 1, i2) == ep_now)

        def to_base(i: torch.Tensor) -> torch.Tensor:
            w = torch.gather(ev_xy, 1, i.unsqueeze(-1).expand(N, B, 2)) - root_xy.transpose(0, 1)
            c, s = torch.cos(yaw.t()), torch.sin(yaw.t())
            return torch.stack([c * w[..., 0] + s * w[..., 1], -s * w[..., 0] + c * w[..., 1]], dim=-1)

        labels = torch.cat([to_base(i1), to_base(i2)], dim=-1)  # (N, B, 4)
        return x.transpose(0, 1)[ok], labels[ok]

    def _refit(self, env: ManagerBasedRLEnv) -> None:
        x, y = self._labels()
        env.fpm_label_rate = x.shape[0] / max(self._B * self.num_envs, 1)
        if x.shape[0] < 1000:
            print(f"[FPM] only {x.shape[0]} labelled states; refit skipped")
            return
        f = self._fit
        perm = torch.randperm(x.shape[0], device=self.device)
        n_train = int(0.8 * x.shape[0])
        tr, va = perm[:n_train], perm[n_train:]
        # rsl_rl collects rollouts under torch.inference_mode; training needs it off, and
        # everything it touches (data, model) created outside it.
        with torch.inference_mode(False), torch.enable_grad():
            xt, yt, xv, yv = x[tr].clone(), y[tr].clone(), x[va].clone(), y[va].clone()
            model = FootholdModel(self._num_inputs).to(self.device)
            with torch.no_grad():
                model.mean.copy_(xt.mean(0))
                model.std.copy_(xt.std(0).clamp(min=1e-3))
            opt = torch.optim.Adam(model.parameters(), lr=f["learning_rate"])
            for _ in range(f["epochs"]):
                for b in torch.randperm(n_train, device=self.device).split(f["batch_size"]):
                    loss = nn.functional.huber_loss(model(xt[b]), yt[b], delta=f["huber_delta"])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                err = (model(xv) - yv).view(-1, 2, 2).norm(dim=-1).mean(0)
                self._fpm.load_state_dict(model.state_dict())
        self._fpm_ready = True
        env.fpm_val_error = (float(err[0]), float(err[1]))
        iteration = env.common_step_counter // self._steps_per_iter
        self._save(env, iteration)
        print(
            f"[FPM] refit at iteration {iteration}: {x.shape[0]} states, "
            f"val error {100 * err[0]:.1f} / {100 * err[1]:.1f} cm"
        )

    # ------------------------------------------------------------------ term
    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        self._map.reset(env_ids)
        tm = getattr(self._env, "termination_manager", None)
        if self._walking or tm is None or self._env.common_step_counter == 0:
            return
        ids = slice(None) if env_ids is None else env_ids
        self._ends += int(tm.dones[ids].sum())
        self._timeouts += int(tm.time_outs[ids].sum())

    def _check_walking(self, env: ManagerBasedRLEnv) -> None:
        """Every ``walk_check_iterations``: the time-out share of the episodes that ended."""
        if self._walking or env.common_step_counter < self._next_walk_check:
            return
        self._next_walk_check = env.common_step_counter + self._walk_check_steps
        rate = self._timeouts / max(self._ends, 1)
        env.fpm_walk_rate = rate
        self._ends = self._timeouts = 0
        if rate >= self._walk_threshold:
            self._walking = True
            print(f"[FPM] walking at iteration {env.common_step_counter // self._steps_per_iter} (time-out {rate:.2f})")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        feet_cfg: SceneEntityCfg,
        contact_cfg: SceneEntityCfg,
        command_name: str,
        map_params: dict,
        patch_size: int = 7,
        bootstrap_footholds: tuple = ((0.0, 0.1), (0.0, -0.1)),
        num_steps_per_env: int = 24,
        bootstrap_iterations: int = 500,
        refit_interval: int = 500,
        walk_threshold: float = 0.8,
        walk_check_iterations: int = 50,
        buffer_steps: int = 150,
        min_air_time: float = 0.05,
        train: bool = True,
        fit: dict | None = None,
        debug_vis: bool = False,
    ) -> torch.Tensor:
        if not self._load_checked and hasattr(env, "fpm_load_dir"):
            self._load_checked = True
            self._try_load(env)
        self._map.update(env)
        x = self._fpm_input(env)

        # one record / refit check per env step (the term may be called again in a reset)
        if self._train and env.common_step_counter != self._last_counter:
            self._last_counter = env.common_step_counter
            self._record(env, x)
            self._check_walking(env)
            if (
                self._walking
                and env.common_step_counter >= self._next_refit_step
                and self._buf_count == self._B
            ):
                self._refit(env)
                self._next_refit_step = env.common_step_counter + self._refit_iters * self._steps_per_iter

        if self._fpm_ready:
            with torch.no_grad():
                footholds = self._fpm(x)
        else:
            footholds = self._bootstrap.expand(self.num_envs, 4)
        out = self._sample(footholds)
        if debug_vis:
            self._visualize(env, out)
        return out

    def _visualize(self, env: ManagerBasedRLEnv, out: torch.Tensor) -> None:
        """Magenta: predicted footholds; cyan: the patch cells at their sampled height."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        if not hasattr(self, "_vis"):
            self._vis = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/FootholdPatches",
                    markers={
                        "foothold": sim_utils.SphereCfg(
                            radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0))
                        ),
                        "cell": sim_utils.SphereCfg(
                            radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0))
                        ),
                    },
                )
            )
        a = env.scene[self._asset_cfg.name]
        root = a.data.root_pos_w
        yaw = _yaw_from_quat(a.data.root_quat_w)
        c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
        kk = self._k * self._k
        pts, kinds = [], []
        lattice = self._patch_points(torch.cat([out[:, :2], out[:, 2 + kk : 4 + kk]], dim=-1))  # (N, 2, kk, 2)
        for i in range(2):
            seg = out[:, i * (2 + kk) : (i + 1) * (2 + kk)]
            p, h = seg[:, :2], seg[:, 2:]
            bx, by = lattice[:, i, :, 0], lattice[:, i, :, 1]
            cells = torch.stack(
                [root[:, :1] + c * bx - s * by, root[:, 1:2] + s * bx + c * by, root[:, 2:3] - h - self._offset], -1
            )
            foot = torch.stack(
                [
                    root[:, 0] + c[:, 0] * p[:, 0] - s[:, 0] * p[:, 1],
                    root[:, 1] + s[:, 0] * p[:, 0] + c[:, 0] * p[:, 1],
                    root[:, 2] - h[:, kk // 2] - self._offset + 0.02,
                ],
                -1,
            )
            pts += [foot.unsqueeze(1), cells]
            kinds += [torch.zeros(self.num_envs, 1, device=self.device), torch.ones(self.num_envs, kk, device=self.device)]
        pts = torch.cat(pts, 1).reshape(-1, 3)
        kinds = torch.cat(kinds, 1).reshape(-1).long()
        self._vis.visualize(translations=pts, marker_indices=kinds)


def fpm_val_error_1(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Validation error of the first predicted foothold at the last refit (m). Logging only."""
    return torch.tensor(getattr(env, "fpm_val_error", (0.0, 0.0))[0], device=env.device)


def fpm_val_error_2(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Validation error of the second predicted foothold at the last refit (m). Logging only."""
    return torch.tensor(getattr(env, "fpm_val_error", (0.0, 0.0))[1], device=env.device)


def fpm_label_rate(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Fraction of buffered states that got a label at the last refit. Logging only."""
    return torch.tensor(getattr(env, "fpm_label_rate", 0.0), device=env.device)


def fpm_walk_rate(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Time-out share of finished episodes at the last walk check (0 once the FPM is on). Logging only."""
    return torch.tensor(getattr(env, "fpm_walk_rate", 0.0), device=env.device)
