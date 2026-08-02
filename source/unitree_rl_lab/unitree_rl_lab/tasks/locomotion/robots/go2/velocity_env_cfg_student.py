"""Student distillation envs, split into two phases after the paper's schedule.

"Learning robust perceptive locomotion for quadrupedal robots in the wild"
(Miki et al., 2022) does not start student distillation on the full training
environment: section S3 puts the student on flat terrain with completely clean
height samples first, brings the terrain curriculum in next, and only then ramps
the height-sample noise up to full magnitude. The two envs here mirror that:

  Student-Phase1  flat terrain, clean height scan -- the student learns to copy
                  the teacher and the belief GRU learns to reconstruct an
                  undistorted height map before anything is taken away from it.
  Student-Phase2  the deployment terrain mix, with the height-scan noise ramping
                  in over ``NOISE_START_ITERATION`` .. ``NOISE_FULL_ITERATION``.

Phase 2 resumes from the Phase 1 checkpoint:

    ./unitree_rl_lab.sh -p scripts/rsl_rl/train.py --task Go2-v3-Student-Phase2 \
        --resume --previous-task Go2-v3-Student-Phase1

Both phases keep Phase 4's rewards, terminations and commands, so terrain and
height-scan noise are the only things that differ between them (and rewards do
not enter the distillation loss at all -- they only drive the terrain and
command curricula).
"""

import copy

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp, terrains
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_distillation_cfg import BeliefDistillationRunnerCfg
from unitree_rl_lab.tasks.locomotion.mdp.height_scan_noise import (
    HeightScanExcludingBodyNoisy,
    HeightScanNoiseCfg,
    HeightScanNoiseConditionCfg,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import CurriculumCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_go2 import (
    CriticCfgGo2,
    ObservationsCfgGo2,
    POLICY_HEIGHT_SCAN_CFG,
    PolicyCfgGo2,
)
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase1 import PHASE1_TERRAIN_CFG
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg_phase4 import (
    RobotEnvCfgPhase4,
    RobotSceneCfgPhase4,
)

# ---------------------------------------------------------------------------
# Noise schedule. The only two numbers to touch when re-timing the ramp.
# ---------------------------------------------------------------------------
NOISE_START_ITERATION = 100
"""Phase 2 iterations with a clean height scan before any noise is added."""
NOISE_FULL_ITERATION = 500
"""Phase 2 iteration at which the conditions below reach full magnitude."""

# Iterations are derived from env steps, so this has to track the runner.
NUM_STEPS_PER_ENV = BeliefDistillationRunnerCfg().num_steps_per_env


# ---------------------------------------------------------------------------
# Mapping conditions (paper S8 / Figure 6B), drawn per episode at 60/30/10.
#
# Values are standard deviations in metres. Where the paper's z vector is
# unambiguous the value is sqrt(z) (it documents z as a variance); where it is
# not -- see the module docstring of mdp/height_scan_noise.py -- the value is
# chosen for this 5 cm body-centered grid instead. Per-condition sources:
#
#   per_point_std     nominal/offset sqrt(z1=0.005), noisy sqrt(z1=0.1)
#   outlier_std       nominal sqrt(z4=0.03), offset sqrt(0.1), noisy sqrt(0.3)
#   outlier_prob      paper z5 verbatim
#   per_step_offset   stand-in for the paper's per-foot, per-step eps_fz; ours is
#                     shared by the whole scan, so it is kept well below the
#                     paper's per-foot figure
#   per_episode_offset stand-in for the paper's per-foot, per-episode w_z, whose
#                     z entry did not survive into the published vector
#
# All magnitude terms (everything but probability/outlier_prob) are then
# halved from those paper-derived numbers: on this project's terrain (wall/box
# heights 5-25 cm), the paper's per-point std alone (7.1 cm, resampled on every
# point every control step) was already close to the terrain's own relief, and
# looked like noise dominating signal in play-mode visualization rather than a
# usable-but-imprecise scan. See visualize_estimate's markers before tuning
# further -- these are a starting point, not a re-derivation from a real sensor.
# ---------------------------------------------------------------------------
NOMINAL_MAPPING = HeightScanNoiseConditionCfg(
    probability=0.60,
    per_point_std=0.0355,
    per_step_offset_std=0.01,
    per_episode_offset_std=0.025,
    outlier_std=0.0865,
    outlier_prob=0.05,
)
"""Normal mapping conditions: the scan is usable, just imprecise."""

LARGE_OFFSET_MAPPING = HeightScanNoiseConditionCfg(
    probability=0.30,
    per_point_std=0.0355,
    per_step_offset_std=0.025,
    per_episode_offset_std=0.10,
    outlier_std=0.158,
    outlier_prob=0.02,
)
"""Pose-estimation drift / deformable ground: the scan is coherent but displaced."""

LARGE_NOISE_MAPPING = HeightScanNoiseConditionCfg(
    probability=0.10,
    per_point_std=0.158,
    per_step_offset_std=0.05,
    per_episode_offset_std=0.10,
    outlier_std=0.274,
    outlier_prob=0.30,
)
"""Occlusion / sensor failure: the scan carries essentially no terrain information."""

STUDENT_HEIGHT_SCAN_NOISE_CFG = HeightScanNoiseCfg(
    nominal=NOMINAL_MAPPING,
    large_offset=LARGE_OFFSET_MAPPING,
    large_noise=LARGE_NOISE_MAPPING,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    start_iteration=NOISE_START_ITERATION,
    full_iteration=NOISE_FULL_ITERATION,
)


def _noisy_height_scan(
    noise_cfg: HeightScanNoiseCfg,
    debug_vis: bool = False,
    debug_vis_env_index: int | None = 0,
) -> ObsTerm:
    """The policy height-scan term, swapped onto the noisy class-based implementation.

    Built by copy rather than ``ObsTerm.replace`` because ``replace`` type-checks
    each field against the value it is overwriting, and ``func`` goes from a plain
    function to a ``ManagerTermBase`` subclass here.
    """
    term = copy.deepcopy(POLICY_HEIGHT_SCAN_CFG)
    term.func = HeightScanExcludingBodyNoisy
    term.params["scan_noise"] = noise_cfg
    term.params["debug_vis_noisy_scan"] = debug_vis
    term.params["debug_vis_env_index"] = debug_vis_env_index
    return term


NOISY_HEIGHT_SCAN_CFG = _noisy_height_scan(STUDENT_HEIGHT_SCAN_NOISE_CFG)

# Play restarts the step counter, so the training ramp would leave the scan clean for
# a whole play session; this variant is already at full magnitude on step 0. It also
# draws the corrupted scan in cyan next to the RayCaster's own markers, so the
# difference between the true terrain and what the student sees is visible.
# debug_vis_env_index=None draws every env (cheap enough at play's 32 envs; training
# keeps the default single index since marking all 4096 envs every step is not).
PLAY_NOISY_HEIGHT_SCAN_CFG = _noisy_height_scan(
    STUDENT_HEIGHT_SCAN_NOISE_CFG.replace(start_iteration=0, full_iteration=0),
    debug_vis=True,
    debug_vis_env_index=None,
)

# ---------------------------------------------------------------------------
# Phase 2 terrain: what the student is expected to handle on deployment. No
# stairs -- the walls are what this lineage actually trains and tests on, and
# stair behaviour carries over through the Phase 4 teacher rather than through
# this mix.
# ---------------------------------------------------------------------------
STUDENT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_cols=20,
    num_rows=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),
        "thin_wall": terrains.MeshThinWallTerrainCfg(
            proportion=0.20,
            wall_height_range=(0.05, 0.25),
            wall_thickness_range=(0.15, 0.05),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
        "floating_thin_wall": terrains.MeshFloatingThinWallTerrainCfg(
            proportion=0.40,
            wall_height_range=(0.05, 0.25),
            wall_thickness_range=(0.15, 0.05),
            wall_spacing=0.60,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)

PLAY_TERRAIN_CFG = STUDENT_TERRAIN_CFG.replace(
    num_cols=3,
    num_rows=5,
    sub_terrains={
        name: cfg
        for name, cfg in STUDENT_TERRAIN_CFG.sub_terrains.items()
        if name in ("boxes", "thin_wall", "floating_thin_wall")
    },
)
"""Play-only terrain: obstacles only (no flat/random_rough), 3 cols x 5 rows."""


@configclass
class TeacherObsCfgGo2(PolicyCfgGo2):
    """Clean proprio|extero for teacher actions and reconstruction targets.

    Same term order / dims as the student's policy group so extero slices align,
    but ``enable_corruption=False`` and an unperturbed height scan.
    """

    height_scan = POLICY_HEIGHT_SCAN_CFG

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class StudentPolicyCfgNoisy(PolicyCfgGo2):
    """Student policy obs: same layout as teacher training, height scan corrupted."""

    height_scan = NOISY_HEIGHT_SCAN_CFG


@configclass
class ObservationsCfgStudentPhase1(ObservationsCfgGo2):
    """Phase 1: student and teacher both see a clean height scan."""

    policy: PolicyCfgGo2 = PolicyCfgGo2()
    teacher: TeacherObsCfgGo2 = TeacherObsCfgGo2()
    critic: CriticCfgGo2 = CriticCfgGo2()


@configclass
class ObservationsCfgStudentPhase2(ObservationsCfgGo2):
    """Phase 2: noisy student height scan, clean teacher / reconstruction target."""

    policy: StudentPolicyCfgNoisy = StudentPolicyCfgNoisy()
    teacher: TeacherObsCfgGo2 = TeacherObsCfgGo2()
    critic: CriticCfgGo2 = CriticCfgGo2()


@configclass
class CurriculumCfgStudentPhase2(CurriculumCfg):
    """Phase 2 adds the noise ramp to the logged curricula."""

    height_scan_noise = CurrTerm(func=mdp.height_scan_noise_level)


@configclass
class RobotSceneCfgStudentPhase1(RobotSceneCfgPhase4):
    terrain = RobotSceneCfgPhase4().terrain.replace(
        terrain_generator=PHASE1_TERRAIN_CFG,
        max_init_terrain_level=0,
    )


@configclass
class RobotSceneCfgStudentPhase2(RobotSceneCfgPhase4):
    terrain = RobotSceneCfgPhase4().terrain.replace(
        terrain_generator=STUDENT_TERRAIN_CFG,
        max_init_terrain_level=5,
    )


@configclass
class RobotEnvCfgStudentPhase1(RobotEnvCfgPhase4):
    """Student phase 1: flat terrain, clean height scan."""

    scene: RobotSceneCfgStudentPhase1 = RobotSceneCfgStudentPhase1(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgStudentPhase1 = ObservationsCfgStudentPhase1()


@configclass
class RobotEnvCfgStudentPhase2(RobotEnvCfgPhase4):
    """Student phase 2: deployment terrain mix, height-scan noise ramping in."""

    scene: RobotSceneCfgStudentPhase2 = RobotSceneCfgStudentPhase2(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfgStudentPhase2 = ObservationsCfgStudentPhase2()
    curriculum: CurriculumCfgStudentPhase2 = CurriculumCfgStudentPhase2()


@configclass
class RobotPlayEnvCfgStudentPhase1(RobotEnvCfgStudentPhase1):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotPlayEnvCfgStudentPhase2(RobotEnvCfgStudentPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator = copy.deepcopy(PLAY_TERRAIN_CFG)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.height_scan = copy.deepcopy(PLAY_NOISY_HEIGHT_SCAN_CFG)
        # Raw RayCaster markers show the clean terrain; off here so only the noisy
        # cyan scan (what the student actually sees) is visible.
        self.scene.height_scanner.debug_vis = False
