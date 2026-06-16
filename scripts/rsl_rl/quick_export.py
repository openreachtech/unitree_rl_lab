import os
import torch
from isaaclab_rl.rsl_rl import export_policy_as_onnx

# ターゲットのパスを設定
run_dir = os.path.expanduser("~/isaacsim/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity_v1/2026-06-06_01-34-08")
checkpoint_path = os.path.join(run_dir, "model_9999.pt")
export_model_dir = os.path.join(run_dir, "exported")

print("🔄 1万回モデル(PyTorch)を読み込み中...")
# チェックポイントのロード
loaded_dict = torch.load(checkpoint_path, map_location="cpu")

# rsl_rl のモデル構造（Actor-Critic）をインポートして再現
from rsl_rl.modules import ActorCritic
# パラメータからモデルの引数を安全に抽出
actor_critic = ActorCritic(
    num_actor_obs=loaded_dict["model_state_dict"]["actor.0.weight"].shape[1],
    num_critic_obs=loaded_dict["model_state_dict"]["critic.0.weight"].shape[1],
    num_actions=loaded_dict["model_state_dict"]["actor.6.weight"].shape[0],
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128]
)
actor_critic.load_state_dict(loaded_dict["model_state_dict"])
policy_nn = actor_critic.actor

print("🚀 ONNX形式への変換を開始します...")
os.makedirs(export_model_dir, exist_ok=True)
export_policy_as_onnx(policy_nn, normalizer=None, path=export_model_dir, filename="policy.onnx")

print(f"✨ 成功! ONNXファイルがここに生成されました:\n{os.path.join(export_model_dir, 'policy.onnx')}")
