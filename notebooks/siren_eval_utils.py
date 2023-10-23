from typing import Dict, List

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import wandb
from omegaconf import DictConfig

from brax.training.acme.running_statistics import RunningStatisticsState
from brax.training.types import Params


def layer_std_sac(stats: RunningStatisticsState, weight_values: jp.ndarray) -> float:
    action_std = jp.ones((weight_values.shape[0] - stats.std.shape[0]))
    full_std = jp.concatenate([stats.std, action_std])
    action_mean = jp.zeros((weight_values.shape[0] - stats.mean.shape[0]))
    full_mean = jp.concatenate([stats.mean, action_mean])
    scaled_weights = (weight_values - full_mean.reshape(-1, 1)) / (
        full_std.reshape(-1, 1)
    )
    effective_frequency_std = jp.std(scaled_weights, axis=1)
    return effective_frequency_std


def layer_std_only_state_input(
    stats: RunningStatisticsState, weight_values: jp.ndarray
) -> float:
    scaled_weights = (weight_values - stats.mean.reshape(-1, 1)) / (
        stats.std.reshape(-1, 1)
    )
    effective_frequency_std = jp.std(scaled_weights)
    return effective_frequency_std


def compute_layer_std_dev_q_params(
    stats: RunningStatisticsState, q_params: Params, sac: bool = False
) -> float:
    def layer_std_for_one_q_network(network_index: int = 0) -> float:
        if sac:
            weight_values = q_params["params"][f"LFFMLP_{network_index}"]["hidden_0"][
                "kernel"
            ]
            return layer_std_sac(stats, weight_values)
        else:
            weight_values = q_params["params"]["LFF_0"]["dense"]["kernel"]
            return layer_std_only_state_input(stats, weight_values)

    if sac:
        overall_std = (
            layer_std_for_one_q_network(network_index=0)
            + layer_std_for_one_q_network(network_index=1)
        ) / 2
    else:
        overall_std = layer_std_for_one_q_network()
    return overall_std


def compute_layer_std_dev_policy_params(
    stats: RunningStatisticsState, policy_params: Params
) -> float:
    weight_values = policy_params["params"]["LFF_0"]["dense"]["kernel"]
    return layer_std_only_state_input(stats, weight_values)


def get_dimension_to_plot(cfg: DictConfig, params: Params) -> jp.ndarray:
    return jax.random.randint(
        jax.random.PRNGKey(cfg["seed"]),
        shape=(15,),
        minval=0,
        maxval=len(params[0].mean) - 1,
    )


def get_points_to_plot(
    mean: jp.ndarray,
    dim: int,
    number_of_points_to_plot: int,
) -> jp.ndarray:
    points_to_plot = []
    dim_mean = 0
    dim_std_dev = 1
    for i in range(1, number_of_points_to_plot):
        diff = (dim_std_dev * i) / (number_of_points_to_plot / 3)
        new_point_plus = jp.zeros_like(mean)
        new_point_plus = new_point_plus.at[dim].set(dim_mean + diff)
        new_point_minus = jp.zeros_like(mean)
        new_point_minus = new_point_minus.at[dim].set(dim_mean - diff)
        points_to_plot.append(new_point_plus)
        points_to_plot.append(new_point_minus)
    return jp.array(points_to_plot)


def get_outputs(points_to_plot: jp.ndarray, params: Params, network: str) -> jp.ndarray:
    if network == "policy":
        return jp.sin(
            jp.matmul(points_to_plot, params[1]["params"]["LFF_0"]["dense"]["kernel"])
            + params[1]["params"]["hidden_0"]["bias"]
        )
    elif network == "value":
        return jp.sin(
            jp.matmul(
                points_to_plot,
                params[2]["params"]["LFFMLP_0"]["LFF_0"]["dense"]["kernel"],
            )
            + params[2]["params"]["hidden_0"]["bias"]
        )
    else:
        raise NotImplementedError("Network not implemented", network)


def plot_neuron(
    dimension: int,
    points_to_plot: jp.ndarray,
    outputs: jp.ndarray,
    params: Params,
    cfg: DictConfig,
) -> None:
    dim_to_plot = jax.random.randint(
        jax.random.PRNGKey(cfg["seed"] + 10),
        shape=(1,),
        minval=0,
        maxval=points_to_plot.shape[1] - 1,
    )
    plt.scatter(points_to_plot[:, dimension], outputs[:, dim_to_plot])
    wandb.log(
        {f"dimension_{dimension}_neuron_{dim_to_plot}_after_sin": wandb.Image(plt)}
    )
    plt.close()


def plot_neuron_activations(cfg: DictConfig, params: Params) -> None:
    number_of_points_to_plot = 20

    dimension_to_plot = get_dimension_to_plot(cfg, params)
    network_list = ["policy"]  # , "value"]
    for network in network_list:
        for dimension in dimension_to_plot:
            points_to_plot = get_points_to_plot(
                mean=params[0].mean,
                dim=dimension,
                number_of_points_to_plot=number_of_points_to_plot,
            )
            outputs = get_outputs(
                points_to_plot=points_to_plot, params=params, network=network
            )
            plot_neuron(
                dimension=dimension,
                points_to_plot=points_to_plot,
                outputs=outputs,
                params=params,
                cfg=cfg,
            )


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def remove_keys_with_substring(d: dict, substring: str) -> dict:
    return {k: v for k, v in d.items() if substring not in k}


def filter_metrics(metrics: List[dict]) -> List[dict]:
    eval_metrics = []

    for dict in metrics:
        new_metrics = remove_keys_with_substring(dict, "training")
        eval_metrics.append(new_metrics)
    return eval_metrics


def dump_metrics_dict_into_csv(metrics: List[dict], cfg: dict) -> None:
    import csv
    from datetime import datetime

    flattened_cfg = flatten_dict(cfg)

    metrics = filter_metrics(metrics)
    keys = metrics[0].keys()

    for data in metrics:
        data.update(flattened_cfg)

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(f"{current_time}.csv", "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(metrics)
