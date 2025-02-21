# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network definitions."""

import dataclasses
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen

from brax.training import types
from brax.training.spectral_norm import SNDense

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden


class SNMLP(linen.Module):
    """MLP module with Spectral Normalization."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = SNDense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden


def default_mlp_init():
    return linen.initializers.uniform(scale=0.05)


def lff_weight_init(scale: float, num_inputs: int):
    return linen.initializers.normal(stddev=scale / num_inputs)


def lff_bias_init():
    return linen.initializers.uniform(scale=2)


class LFF(linen.Module):
    num_output_features: int
    num_input_features: int
    scale: float

    def setup(self):
        self.dense = linen.Dense(
            features=self.num_output_features,
            kernel_init=lff_weight_init(
                scale=self.scale, num_inputs=self.num_input_features
            ),
            bias_init=lff_bias_init(),
        )

    def __call__(self, x):
        return jnp.sin(jnp.pi * (self.dense(x) - 1))


class LFFMLP(linen.Module):
    layer_sizes: Sequence[int]
    scale: float

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes[:-1]):
            if i == 0:
                hidden = LFF(
                    num_input_features=self.layer_sizes[i],
                    num_output_features=self.layer_sizes[i + 1],
                    scale=self.scale,
                    name=f"LFF_{i}",
                )(hidden)
            if i == len(self.layer_sizes) - 2:
                hidden = linen.Dense(
                    self.layer_sizes[i + 1],
                    name=f"hidden_{i}",
                    kernel_init=default_mlp_init(),
                    use_bias=True,
                )(hidden)
            else:
                hidden = linen.Dense(
                    self.layer_sizes[i + 1],
                    name=f"hidden_{i}",
                    kernel_init=default_mlp_init(),
                    use_bias=True,
                )(hidden)
                hidden = linen.relu(hidden)
        return hidden


class LinearFirstLayerMLP(linen.Module):
    layer_sizes: Sequence[int]
    scale: float

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes[:-1]):
            if i == len(self.layer_sizes) - 2 or i == 0:
                hidden = linen.Dense(
                    self.layer_sizes[i + 1],
                    name=f"hidden_{i}",
                    kernel_init=default_mlp_init(),
                    use_bias=True,
                )(hidden)
            else:
                hidden = linen.Dense(
                    self.layer_sizes[i + 1],
                    name=f"hidden_{i}",
                    kernel_init=default_mlp_init(),
                    use_bias=True,
                )(hidden)
                hidden = linen.relu(hidden)
        return hidden


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    use_lff: Optional[bool] = None,
    lff_scale: Optional[float] = None,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    if use_lff:
        policy_module = LFFMLP(
            layer_sizes=[40 * obs_size] + list(hidden_layer_sizes) + [param_size],
            scale=lff_scale,
        )
    else:
        policy_module = MLP(
            layer_sizes=[40 * obs_size] + list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply
    )


def make_value_network(
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    use_lff: Optional[bool] = None,
    lff_scale: Optional[float] = None,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    if use_lff:
        value_module = LFFMLP(
            layer_sizes=[40 * obs_size] + list(hidden_layer_sizes) + [1],
            scale=lff_scale,
        )
    else:
        value_module = MLP(
            layer_sizes=list(hidden_layer_sizes) + [1],
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs), apply=apply
    )


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    use_lff: bool = False,
    lff_scale: Optional[float] = None,
) -> FeedForwardNetwork:
    """Creates a value network."""

    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                if use_lff:
                    q = LFFMLP(
                        layer_sizes=[40 * hidden.shape[-1]]
                        + list(hidden_layer_sizes)
                        + [1],
                        scale=lff_scale,
                    )(hidden)
                else:
                    q = MLP(
                        list(hidden_layer_sizes) + [1],
                        activation=activation,
                        kernel_init=jax.nn.initializers.lecun_uniform(),
                    )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_model(
    layer_sizes: Sequence[int],
    obs_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    spectral_norm: bool = False,
) -> FeedForwardNetwork:
    """Creates a model.

    Args:
      layer_sizes: layers
      obs_size: size of an observation
      activation: activation
      spectral_norm: whether to use a spectral normalization (default: False).

    Returns:
      a model
    """
    warnings.warn(
        "make_model is deprecated, use make_{policy|q|value}_network instead."
    )
    dummy_obs = jnp.zeros((1, obs_size))
    if spectral_norm:
        module = SNMLP(layer_sizes=layer_sizes, activation=activation)
        model = FeedForwardNetwork(
            init=lambda rng1, rng2: module.init(
                {"params": rng1, "sing_vec": rng2}, dummy_obs
            ),
            apply=module.apply,
        )
    else:
        module = MLP(layer_sizes=layer_sizes, activation=activation)
        model = FeedForwardNetwork(
            init=lambda rng: module.init(rng, dummy_obs), apply=module.apply
        )
    return model


def make_models(
    policy_params_size: int, obs_size: int
) -> Tuple[FeedForwardNetwork, FeedForwardNetwork]:
    """Creates models for policy and value functions.

    Args:
      policy_params_size: number of params that a policy network should generate
      obs_size: size of an observation

    Returns:
      a model for policy and a model for value function
    """
    warnings.warn(
        "make_models is deprecated, use make_{policy|q|value}_network instead."
    )
    policy_model = make_model([32, 32, 32, 32, policy_params_size], obs_size)
    value_model = make_model([256, 256, 256, 256, 256, 1], obs_size)
    return policy_model, value_model
