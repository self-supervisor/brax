import functools
import os
from datetime import datetime
from typing import Dict

import flax
import hydra
import jax
import matplotlib.pyplot as plt
import numpy as onp
import wandb
from jax import numpy as jp
from omegaconf import DictConfig
from siren_eval_utils import (
    compute_layer_std_dev_policy_params,
    compute_layer_std_dev_q_params,
    plot_neuron_activations,
    dump_metrics_dict_into_csv,
)

import brax
from brax import envs
from brax.io import html, json, model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

metrics_dump = []


def get_kwargs_ready(cfg: DictConfig) -> Dict:
    kwargs = dict(cfg)
    kwargs.pop("env_name")
    kwargs.pop("backend")
    kwargs.pop("training_algo")
    kwargs.pop("max_y")
    kwargs.pop("min_y")
    kwargs.pop("wandb_mode")
    return kwargs


@hydra.main(config_path="conf")
def main(cfg: DictConfig):
    wandb.init(
        project="sac_siren_brax",
        config=dict(cfg),
        settings=wandb.Settings(mode=cfg.wandb_mode),
    )
    env = envs.get_environment(env_name=cfg.env_name, backend=cfg.backend)
    trainer = sac.train if cfg.training_algo == "sac" else ppo.train
    kwargs = get_kwargs_ready(cfg)
    train_fn = functools.partial(trainer, **kwargs)

    def progress(num_steps, metrics):
        metrics = {k: onp.array(v) for k, v in metrics.items()}
        metrics_dump.append(metrics)
        wandb.log(metrics, step=num_steps)

    make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)
    dump_metrics_dict_into_csv(metrics=metrics_dump, cfg=cfg)
    if cfg.use_lff == True:
        if cfg.training_algo == "sac":
            std_dev_critic = compute_layer_std_dev_q_params(
                params[0], params[2], sac=True
            )
        else:
            std_dev_critic = compute_layer_std_dev_q_params(
                params[0], params[2], sac=False
            )
        std_dev_actor = compute_layer_std_dev_policy_params(params[0], params[1])
        wandb.log({"std_dev_critic": std_dev_critic, "std_dev_actor": std_dev_actor})
        wandb.log(
            {
                "std_dev_critic_cycles": std_dev_critic / (2 * onp.pi),
                "std_dev_actor_cycles": std_dev_actor / (2 * onp.pi),
            }
        )

    model.save_params("rl_params", params)
    params = model.load_params("rl_params")
    policy_params = (params[0], params[1])
    inference_fn = make_inference_fn(policy_params)
    # plot_neuron_activations(cfg, params)

    env = envs.create(env_name=cfg.env_name, backend=cfg.backend)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    for _ in range(1000):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act, rng)

    html_str = html.render(env.sys.replace(dt=env.dt), rollout)
    wandb.log({"trained_agent_html": wandb.Html(html_str)})


if __name__ == "__main__":
    main()
