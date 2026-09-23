"""A GRU policy with a height map fed in beside it.

The blind lineage's actor is ``obs -> GRU -> MLP``. This adds one input: an elevation
grid, squashed by a small MLP encoder and handed to the actor MLP *next to* the GRU's
output rather than through it::

    height map ---> g_e ---> l_e ---+
                                    |
    proprio ------> GRU ----------> +--> actor MLP --> action

That placement is a deliberate choice rather than the only one. Miki et al.'s teacher
(``doc/papers/Learning_robust_perceptive_locomotion_for_quadrupedal_robots_in_the_wild.md``,
"Policy architecture") encodes the height samples with an MLP and concatenates the latent
with proprioception into one feed-forward network -- no recurrence over the map at all.
Their *student* does put it through a recurrent belief encoder, with a learned sigmoid gate
deciding how much of the map to let past the recurrence; that machinery exists to survive
snow, water, reflections and pose drift. This robot works indoors on a factory floor, where
the delivered map was measured at 2 mm per-cell standard deviation, so the gate would sit
open and the belief state would have little to estimate. What is left after dropping it is
the teacher's arrangement with the existing GRU kept on the proprioceptive branch, which is
what this module implements.

Three arrangements are reachable, because the choice is one concatenation either way and
the cheap thing is to be able to measure it rather than argue about it:

===================  ====================  ================  ==================
``extero_to_memory``  ``extero_skip``      map reaches GRU   map reaches actor
===================  ====================  ================  ==================
True                 False                 yes               only via the GRU
False                True  *(default)*     no                directly
True                 True                  yes               both ways
===================  ====================  ================  ==================

The default keeps the GRU's input width at the blind policy's, so a blind checkpoint's
recurrent weights load into it unchanged and only the actor MLP's first layer is new.

Two further notes on why the encoder is here at all. It is not deference to the map: at
609 cells it is *cheaper* than feeding the grid to the actor MLP raw (609 -> 160 -> 96
then 352 x 512 is 293k parameters against 865 x 512 = 443k), and a 5 cm grid's
neighbouring cells are nearly redundant, so the bottleneck costs little. And the encoder
is a plain MLP over the flattened grid, not per-foot as in the paper -- this robot's
exteroception is a body-centred grid rather than rings around each foot, so there is no
per-foot structure to preserve.

The critic is built the same way from its own observation groups, so it can be given the
noise-free grid while the actor gets the sensor's.
"""

from __future__ import annotations

import copy
import os
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn, Sequence

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState, Memory
from rsl_rl.utils import unpad_trajectories


class ActorCriticPerceptiveRecurrent(nn.Module):
    """``ActorCriticRecurrent`` with an exteroceptive branch around the recurrence.

    Which observation groups are exteroceptive is named explicitly by
    ``extero_obs_groups`` rather than inferred from width, so that adding another
    proprioceptive group later cannot silently reroute it through the encoder. A group
    listed there but absent from a side's observation set simply does not apply to that
    side -- that is how the actor reads the sensor's grid while the critic reads the
    clean one.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        extero_obs_groups: Sequence[str] = (),
        extero_encoder_dims: Sequence[int] = (160,),
        extero_latent_dim: int = 96,
        extero_to_memory: bool = False,
        extero_skip: bool = True,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticPerceptiveRecurrent.__init__ got unexpected arguments, which will be"
                " ignored: " + str(kwargs.keys())
            )
        super().__init__()

        extero_obs_groups = tuple(extero_obs_groups)
        if extero_obs_groups and not (extero_to_memory or extero_skip):
            raise ValueError(
                "extero_obs_groups was given but both extero_to_memory and extero_skip are"
                " False, so the height map would be built and then dropped."
            )
        # Kept because rsl-rl's own actor-critics all expose it under this name.
        self.obs_groups = obs_groups
        self.extero_to_memory = extero_to_memory
        self.extero_skip = extero_skip

        self.actor_proprio_groups, self.actor_extero_groups = self._split_groups(
            obs_groups["policy"], extero_obs_groups
        )
        self.critic_proprio_groups, self.critic_extero_groups = self._split_groups(
            obs_groups["critic"], extero_obs_groups
        )
        for side, proprio_groups in (
            ("policy", self.actor_proprio_groups),
            ("critic", self.critic_proprio_groups),
        ):
            if not proprio_groups:
                raise ValueError(
                    f"Every observation group in the '{side}' set is listed in"
                    " extero_obs_groups; the recurrent branch would have no input."
                )

        actor_proprio_dim = self._group_dim(obs, self.actor_proprio_groups)
        actor_extero_dim = self._group_dim(obs, self.actor_extero_groups)
        critic_proprio_dim = self._group_dim(obs, self.critic_proprio_groups)
        critic_extero_dim = self._group_dim(obs, self.critic_extero_groups)

        # -- exteroceptive encoders ------------------------------------------------
        self.actor_extero_encoder = self._make_encoder(
            actor_extero_dim, extero_encoder_dims, extero_latent_dim, activation
        )
        self.critic_extero_encoder = self._make_encoder(
            critic_extero_dim, extero_encoder_dims, extero_latent_dim, activation
        )
        actor_latent = extero_latent_dim if self.actor_extero_encoder is not None else 0
        critic_latent = extero_latent_dim if self.critic_extero_encoder is not None else 0
        # ``MLP`` is a bare ``nn.Sequential``, so it carries no record of its own widths;
        # the exporters need them, so keep them here.
        self.actor_proprio_dim = actor_proprio_dim
        self.actor_extero_dim = actor_extero_dim

        # -- recurrent branches ----------------------------------------------------
        self.memory_a = Memory(
            actor_proprio_dim + (actor_latent if extero_to_memory else 0),
            rnn_hidden_dim,
            rnn_num_layers,
            rnn_type,
        )
        self.memory_c = Memory(
            critic_proprio_dim + (critic_latent if extero_to_memory else 0),
            rnn_hidden_dim,
            rnn_num_layers,
            rnn_type,
        )

        # -- heads -----------------------------------------------------------------
        self.actor = MLP(
            rnn_hidden_dim + (actor_latent if extero_skip else 0),
            num_actions,
            list(actor_hidden_dims),
            activation,
        )
        self.critic = MLP(
            rnn_hidden_dim + (critic_latent if extero_skip else 0),
            1,
            list(critic_hidden_dims),
            activation,
        )
        print(f"Actor extero encoder: {self.actor_extero_encoder}")
        print(f"Actor RNN: {self.memory_a}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic extero encoder: {self.critic_extero_encoder}")
        print(f"Critic RNN: {self.memory_c}")
        print(f"Critic MLP: {self.critic}")

        # -- observation normalization ---------------------------------------------
        # One flag per side, two normalizers: EmpiricalNormalization is per-dimension, so
        # splitting it across the proprioceptive and exteroceptive halves is equivalent to
        # normalizing the concatenation, and it keeps the halves separable for export.
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = self._make_normalizer(actor_proprio_dim, actor_obs_normalization)
        self.actor_extero_normalizer = self._make_normalizer(actor_extero_dim, actor_obs_normalization)
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = self._make_normalizer(critic_proprio_dim, critic_obs_normalization)
        self.critic_extero_normalizer = self._make_normalizer(critic_extero_dim, critic_obs_normalization)

        # -- action noise ----------------------------------------------------------
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(
                f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'"
            )

        self.distribution = None
        Normal.set_default_validate_args(False)

    # -- construction helpers ------------------------------------------------------

    @staticmethod
    def _split_groups(
        groups: list[str], extero_obs_groups: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        proprio = [g for g in groups if g not in extero_obs_groups]
        extero = [g for g in groups if g in extero_obs_groups]
        return proprio, extero

    @staticmethod
    def _group_dim(obs: TensorDict, groups: list[str]) -> int:
        total = 0
        for group in groups:
            assert len(obs[group].shape) == 2, (
                "ActorCriticPerceptiveRecurrent only supports 1D observations; group"
                f" '{group}' has shape {tuple(obs[group].shape)}."
            )
            total += obs[group].shape[-1]
        return total

    @staticmethod
    def _make_encoder(
        input_dim: int, hidden_dims: Sequence[int], latent_dim: int, activation: str
    ) -> MLP | None:
        if input_dim == 0:
            return None
        return MLP(input_dim, latent_dim, list(hidden_dims), activation)

    @staticmethod
    def _make_normalizer(dim: int, enabled: bool) -> nn.Module:
        if enabled and dim > 0:
            return EmpiricalNormalization(dim)
        return nn.Identity()

    # -- forward helpers -----------------------------------------------------------

    @staticmethod
    def _cat(obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[group] for group in groups], dim=-1)

    def _branch(
        self,
        obs: TensorDict,
        proprio_groups: list[str],
        extero_groups: list[str],
        proprio_normalizer: nn.Module,
        extero_normalizer: nn.Module,
        encoder: MLP | None,
        memory: Memory,
        masks: torch.Tensor | None,
        hidden_state: HiddenState,
    ) -> torch.Tensor:
        """Run one side and return what its head consumes."""
        proprio = proprio_normalizer(self._cat(obs, proprio_groups))
        latent = None
        if encoder is not None:
            latent = encoder(extero_normalizer(self._cat(obs, extero_groups)))

        memory_in = proprio
        if latent is not None and self.extero_to_memory:
            memory_in = torch.cat([proprio, latent], dim=-1)
        out = memory(memory_in, masks, hidden_state)
        if masks is None:
            # Inference mode returns (1, N, H); batch mode is already unpadded to
            # (T, N, H) and has no leading axis to drop.
            out = out.squeeze(0)

        if latent is not None and self.extero_skip:
            if masks is not None:
                # The memory unpads its own output, so the skip has to be unpadded too
                # or the two no longer describe the same timesteps.
                latent = unpad_trajectories(latent, masks)
            out = torch.cat([out, latent], dim=-1)
        return out

    def _update_distribution(self, features: torch.Tensor) -> None:
        mean = self.actor(features)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    # -- rsl-rl interface ----------------------------------------------------------

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def act(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        features = self._actor_features(obs, masks, hidden_state)
        self._update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.actor(self._actor_features(obs, None, None))

    def evaluate(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        features = self._branch(
            obs,
            self.critic_proprio_groups,
            self.critic_extero_groups,
            self.critic_obs_normalizer,
            self.critic_extero_normalizer,
            self.critic_extero_encoder,
            self.memory_c,
            masks,
            hidden_state,
        )
        return self.critic(features)

    def _actor_features(
        self, obs: TensorDict, masks: torch.Tensor | None, hidden_state: HiddenState
    ) -> torch.Tensor:
        return self._branch(
            obs,
            self.actor_proprio_groups,
            self.actor_extero_groups,
            self.actor_obs_normalizer,
            self.actor_extero_normalizer,
            self.actor_extero_encoder,
            self.memory_a,
            masks,
            hidden_state,
        )

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Proprioception then height map -- the layout the exporters flatten to.

        Nothing in rsl-rl calls this (this class overrides ``act``/``evaluate`` and runs
        the branches itself), but it is the one place that states the order the deploy
        side has to pack, and the exporters are written against it.
        """
        return self._cat(obs, self.actor_proprio_groups + self.actor_extero_groups)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self._cat(obs, self.actor_proprio_groups))
            if self.actor_extero_groups:
                self.actor_extero_normalizer.update(self._cat(obs, self.actor_extero_groups))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self._cat(obs, self.critic_proprio_groups))
            if self.critic_extero_groups:
                self.critic_extero_normalizer.update(self._cat(obs, self.critic_extero_groups))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


class _PerceptiveExporterBase(nn.Module):
    """Shared plumbing for the jit / ONNX wrappers.

    isaaclab_rl's generic recurrent exporter assumes ``obs -> memory_a.rnn -> actor``.
    That is wrong here whenever the height map reaches the actor without passing through
    the GRU, which is the default, so exporting through it would silently produce a
    network with the skip connection missing. These wrappers run the real branch.

    The exported signature is one flat observation vector laid out as
    ``[proprioception | height map]`` -- the order
    :meth:`ActorCriticPerceptiveRecurrent.get_actor_obs` concatenates, and the order the
    deploy side has to pack.
    """

    def __init__(self, policy: ActorCriticPerceptiveRecurrent):
        super().__init__()
        if policy.actor_extero_encoder is None:
            raise ValueError(
                "This exporter is for a policy with an exteroceptive branch; the actor here"
                " has none, so isaaclab_rl's generic recurrent exporter already fits it."
            )
        self.proprio_dim = int(policy.actor_proprio_dim)
        self.extero_dim = int(policy.actor_extero_dim)
        self.extero_to_memory = bool(policy.extero_to_memory)
        self.extero_skip = bool(policy.extero_skip)

        self.obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.extero_normalizer = copy.deepcopy(policy.actor_extero_normalizer)
        self.extero_encoder = copy.deepcopy(policy.actor_extero_encoder)
        self.rnn = copy.deepcopy(policy.memory_a.rnn)
        self.actor = copy.deepcopy(policy.actor)
        self.eval()

        self.num_layers = self.rnn.num_layers
        self.hidden_size = self.rnn.hidden_size
        if type(self.rnn).__name__.lower() != "gru":
            raise NotImplementedError(
                f"Only GRU export is implemented; got {type(self.rnn).__name__}."
            )

    def _act(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proprio = self.obs_normalizer(x[..., : self.proprio_dim])
        latent = self.extero_encoder(self.extero_normalizer(x[..., self.proprio_dim :]))

        memory_in = proprio
        if self.extero_to_memory:
            memory_in = torch.cat([proprio, latent], dim=-1)
        out, hidden = self.rnn(memory_in.unsqueeze(0), hidden)
        out = out.squeeze(0)
        if self.extero_skip:
            out = torch.cat([out, latent], dim=-1)
        return self.actor(out), hidden


class PerceptivePolicyJitExporter(_PerceptiveExporterBase):
    """TorchScript wrapper. See :class:`_PerceptiveExporterBase`."""

    def __init__(self, policy: ActorCriticPerceptiveRecurrent):
        super().__init__(policy)
        self.register_buffer("hidden_state", torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        action, hidden = self._act(x, self.hidden_state)
        # ``copy_`` is differentiable in its source, so storing the graph-carrying hidden
        # state would make the buffer require grad -- which then makes ``reset``'s
        # in-place zero illegal, and keeps every step's graph alive across the next.
        self.hidden_state[:] = hidden.detach()
        return action

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0

    def export(self, path: str, filename: str = "policy.pt") -> None:
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        torch.jit.script(self).save(os.path.join(path, filename))


class PerceptivePolicyOnnxExporter(_PerceptiveExporterBase):
    """ONNX wrapper, hidden state in and out. See :class:`_PerceptiveExporterBase`."""

    def __init__(self, policy: ActorCriticPerceptiveRecurrent, verbose: bool = False):
        super().__init__(policy)
        self.verbose = verbose

    def forward(self, x_in: torch.Tensor, h_in: torch.Tensor):
        return self._act(x_in, h_in)

    def export(self, path: str, filename: str = "policy.onnx") -> None:
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.proprio_dim + self.extero_dim)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        torch.onnx.export(
            self,
            (obs, h_in),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,  # matches isaaclab_rl's exporter
            verbose=self.verbose,
            input_names=["obs", "h_in"],
            output_names=["actions", "h_out"],
            dynamic_axes={},
        )
