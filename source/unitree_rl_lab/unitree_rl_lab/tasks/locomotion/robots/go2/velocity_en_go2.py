from isaaclab.envs import ManagerBasedRLEnv


class ManagerBasedRLEnvGo2(ManagerBasedRLEnv):
    """Go2 velocity env; manual curriculum level lives on ``cfg.curriculum.current_level``."""
