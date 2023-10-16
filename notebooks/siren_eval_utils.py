from typing import Dict

import jax.numpy as jnp

from brax.training.acme.running_statistics import RunningStatisticsState


def layer_std(stats: RunningStatisticsState, weight_values: jnp.ndarray) -> float:
    action_std = jnp.ones((weight_values.shape[0] - stats.std.shape[0]))
    full_std = jnp.concatenate([stats.std, action_std])
    action_mean = jnp.zeros((weight_values.shape[0] - stats.mean.shape[0]))
    full_mean = jnp.concatenate([stats.mean, action_mean])
    scaled_weights = (weight_values - full_mean.reshape(-1, 1)) / (
        full_std.reshape(-1, 1)
    )
    effective_frequency_std = jnp.std(scaled_weights, axis=1)
    return effective_frequency_std


def compute_layer_std_dev_q_params(
    stats: RunningStatisticsState, q_params: Dict, sac: bool = False
) -> float:
    """Compute the standard deviation of the layer weights."""

    def layer_std_for_one_q_network(network_index: int = 0) -> float:
        if sac:
            weight_values = q_params["params"][f"MLP_{network_index}"]["hidden_0"][
                "kernel"
            ]
        else:
            weight_values = q_params["params"]["hidden_0"]["kernel"]
        return layer_std(stats, weight_values)

    if sac:
        overall_std = (
            layer_std_for_one_q_network(network_index=0)
            + layer_std_for_one_q_network(network_index=1)
        ) / 2
    else:
        overall_std = layer_std_for_one_q_network()
    return overall_std


def compute_layer_std_dev_policy_params(
    stats: RunningStatisticsState, policy_params
) -> float:
    weight_values = policy_params["params"]["hidden_0"]["kernel"]
    return layer_std(stats, weight_values)
