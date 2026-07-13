import copy
import os
import torch
from typing import Optional

from unitree_rl_lab.assets.models.modules.base_nn import BaseNet


class StudentPolicy(BaseNet):
    """
    Student (deployment) policy with belief encoder / decoder.

    Architecture from: "Learning robust perceptive locomotion for quadrupedal robots
    in the wild" (Miki et al., 2022 / Zhuang et al. follow-ups).

    Unlike the teacher, the student has no privileged encoder. Instead it maintains a
    recurrent belief state that fuses proprioception with (possibly noisy) exteroception,
    and optionally reconstructs clean exteroception via the belief decoder.

    Inputs:
        - proprio_state : command + proprioception          (o_t^p)
        - extero_state  : height-scan (optionally noisy)    (o_t^e)
        - hidden_state  : GRU hidden state from previous step

    Outputs:
        - action                 : deterministic action (distilled from teacher)
        - recurrent_hidden       : next GRU hidden state
        - belief_state           : gated belief latent fed to the base net
        - estimated_extero_state : reconstructed exteroception (if use_decoder)
    """

    class BeliefEncoder(BaseNet):
        def __init__(self, model_cfg):
            super().__init__(model_config=model_cfg)

        def forward(
            self,
            proprio_state: torch.Tensor,
            encoded_extero_state: torch.Tensor,
            hidden_state: Optional[torch.Tensor] = None,
        ):
            """
            :param proprio_state:         [(*seq, proprio_dim)]
            :param encoded_extero_state:  [(*seq, extero_encoder_output)]
            :param hidden_state:          [(num_layers, *batch, hidden)] or None
            :return dict:
                recurrent_output  [(*seq, hidden)]
                recurrent_hidden  [(num_layers, *batch, hidden)]
                belief_state      [(*seq, extero_encoder_output)]
            """
            fused_state = torch.cat((proprio_state, encoded_extero_state), dim=-1)

            if hidden_state is None:
                recurrent_output, recurrent_hidden = self.recurrent_encoder(fused_state)
            else:
                recurrent_output, recurrent_hidden = self.recurrent_encoder(
                    fused_state, hidden_state
                )

            # Gate the single height-map latent with an attention mask from the GRU output.
            belief_state = (
                self.state_encoder(recurrent_output)
                + torch.sigmoid(self.attention_encoder(recurrent_output))
                * encoded_extero_state
            )
            return {
                "recurrent_output": recurrent_output,
                "recurrent_hidden": recurrent_hidden,
                "belief_state": belief_state,
            }

    class BeliefDecoder(BaseNet):
        def __init__(self, model_cfg):
            super().__init__(model_config=model_cfg)

        def forward(self, extero_state, recurrent_output):
            """
            Reconstruct clean exteroception from the GRU output, gated by the raw scan.

            :param extero_state:      [(*seq, extero_dim)]
            :param recurrent_output:  [(*seq, hidden)]  (GRU last output == hidden for GRU)
            :return: estimated_extero_state [(*seq, extero_dim)]
            """
            return (
                self.extero_decoder(recurrent_output)
                + torch.sigmoid(self.attention_encoder(recurrent_output)) * extero_state
            )

    def __init__(self, args, model_cfg):
        self.proprio_dim = args.proprio_obs_dim
        self.extero_dim = args.extero_obs_dim
        self.action_dim = args.action_dim
        self.model_cfg = model_cfg
        self._adapt(args)

        super().__init__(model_config=model_cfg["policy"])
        self.belief_encoder = self.BeliefEncoder(model_cfg["belief_encoder"])
        self.belief_decoder = self.BeliefDecoder(model_cfg["belief_decoder"])

    def _adapt(self, args):
        """Inject input/output dimensions into the config before BaseNet builds the layers."""
        policy_mlp = self.model_cfg["policy"]["MLP"]
        belief_enc = self.model_cfg["belief_encoder"]
        belief_dec = self.model_cfg["belief_decoder"]

        # Single Go2 lidar height-map encoder (not per-leg).
        policy_mlp["extero_encoder"]["input"] = self.extero_dim
        extero_latent_dim = policy_mlp["extero_encoder"]["output"]

        policy_mlp["base_net"]["input"] = self.proprio_dim + extero_latent_dim
        policy_mlp["base_net"]["output"] = self.action_dim

        # Belief encoder: GRU over [proprio | encoded_extero], then attention / state MLPs.
        belief_enc["GRU"]["recurrent_encoder"]["input"] = self.proprio_dim + extero_latent_dim
        gru_hidden = belief_enc["GRU"]["recurrent_encoder"]["hidden"]

        belief_enc["MLP"]["attention_encoder"]["input"] = gru_hidden
        belief_enc["MLP"]["attention_encoder"]["output"] = extero_latent_dim
        belief_enc["MLP"]["state_encoder"]["input"] = gru_hidden
        # state_encoder.output matches extero_latent_dim (default 24)

        # Belief decoder: reconstruct full extero scan from GRU output.
        belief_dec["MLP"]["attention_encoder"]["input"] = gru_hidden
        belief_dec["MLP"]["attention_encoder"]["output"] = self.extero_dim
        belief_dec["MLP"]["extero_decoder"]["input"] = gru_hidden
        belief_dec["MLP"]["extero_decoder"]["output"] = self.extero_dim

    @property
    @torch.jit.unused
    def gru_num_layers(self):
        """Python-only introspection (export-wrapper construction); not needed at inference
        time, so it's excluded from scripting -- `self.model_cfg` is a heterogeneous dict
        TorchScript can't type."""
        return self.model_cfg["belief_encoder"]["GRU"]["recurrent_encoder"]["num_layers"]

    @property
    @torch.jit.unused
    def gru_hidden_size(self):
        """See ``gru_num_layers``."""
        return self.model_cfg["belief_encoder"]["GRU"]["recurrent_encoder"]["hidden"]

    def forward(
        self,
        proprio_state,
        extero_state,
        hidden_state: Optional[torch.Tensor] = None,
        use_decoder: bool = True,
    ):
        """
        :param proprio_state: [(*seq, proprio_dim)]   seq = (L,) or (L, N) with batch_first=False
        :param extero_state:  [(*seq, extero_dim)]
        :param hidden_state:  [(num_layers, *batch, hidden)] or None
        :param use_decoder:   whether to run the belief decoder (needed for reconstruction loss)
        :return dict with action, recurrent_hidden, belief_state, and optionally estimated_extero_state
        """
        encoded_extero = self.extero_encoder(extero_state)

        belief_out = self.belief_encoder(proprio_state, encoded_extero, hidden_state)
        fused = torch.cat((proprio_state, belief_out["belief_state"]), dim=-1)

        output = {
            "action": self.base_net(fused),
            "recurrent_hidden": belief_out["recurrent_hidden"],
            "belief_state": belief_out["belief_state"],
        }
        if use_decoder:
            output["estimated_extero_state"] = self.belief_decoder(
                extero_state, belief_out["recurrent_output"]
            )
        return output

    def get_action(self, proprio_state, extero_state, hidden_state):
        """
        Single-step inference for rollout / deployment (no decoder).

        :param proprio_state: [(N, proprio_dim)]
        :param extero_state:  [(N, extero_dim)]
        :param hidden_state:  [(num_layers, N, hidden)]
        :return: action [(N, action_dim)], next_hidden [(num_layers, N, hidden)]
        """
        # GRU expects a leading sequence dim when batch_first=False.
        proprio_state = proprio_state.unsqueeze(0)
        extero_state = extero_state.unsqueeze(0)
        out = self.forward(proprio_state, extero_state, hidden_state, use_decoder=False)
        return out["action"].squeeze(0), out["recurrent_hidden"]


class StudentPolicyJitExporter(torch.nn.Module):
    """TorchScript export wrapper for a trained ``StudentPolicy``.

    isaaclab_rl's generic recurrent exporter (``isaaclab_rl.rsl_rl.export_policy_as_jit``)
    assumes a policy that decomposes into a separate ``memory_s.rnn`` (raw RNN) plus a
    stateless MLP ``student`` consuming just the RNN output (``policy.memory_s.rnn`` /
    ``policy.student(x)``). ``StudentPolicy`` is a single fused module (extero encoder ->
    belief GRU + attention gating -> base net) that doesn't decompose that way, so it
    can't be passed through that exporter (``StudentTeacher`` has no ``memory_s``). This
    wraps ``get_action`` instead, matching the *external* conventions of that exporter
    (buffer-based hidden state + ``reset()`` for JIT) so downstream deploy code sees the
    same shape it would for any other recurrent policy.
    """

    def __init__(self, student_policy: "StudentPolicy", normalizer=None):
        super().__init__()
        self.student = copy.deepcopy(student_policy)
        self.student.eval()
        self.proprio_dim = student_policy.proprio_dim
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()
        self.register_buffer(
            "hidden_state",
            torch.zeros(student_policy.gru_num_layers, 1, student_policy.gru_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        proprio = x[..., : self.proprio_dim]
        extero = x[..., self.proprio_dim :]
        action, hidden = self.student.get_action(proprio, extero, self.hidden_state)
        self.hidden_state[:] = hidden
        return action

    @torch.jit.export
    def reset(self):
        self.hidden_state[:] = 0.0

    def export(self, path, filename="policy.pt"):
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        scripted = torch.jit.script(self)
        scripted.save(os.path.join(path, filename))


class StudentPolicyOnnxExporter(torch.nn.Module):
    """ONNX export wrapper for a trained ``StudentPolicy``. See ``StudentPolicyJitExporter``."""

    def __init__(self, student_policy: "StudentPolicy", normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.student = copy.deepcopy(student_policy)
        self.student.eval()
        self.proprio_dim = student_policy.proprio_dim
        self.extero_dim = student_policy.extero_dim
        self.num_layers = student_policy.gru_num_layers
        self.hidden_size = student_policy.gru_hidden_size
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

    def forward(self, x_in: torch.Tensor, h_in: torch.Tensor):
        x_in = self.normalizer(x_in)
        proprio = x_in[..., : self.proprio_dim]
        extero = x_in[..., self.proprio_dim :]
        return self.student.get_action(proprio, extero, h_in)

    def export(self, path, filename="policy.onnx"):
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()
        opset_version = 18  # matches isaaclab_rl's exporter (linux-aarch compatibility)
        obs = torch.zeros(1, self.proprio_dim + self.extero_dim)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        torch.onnx.export(
            self,
            (obs, h_in),
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs", "h_in"],
            output_names=["actions", "h_out"],
            dynamic_axes={},
        )
