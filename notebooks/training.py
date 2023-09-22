import functools
import jax
import os

os.environ["HYDRA_FULL_ERROR"] = "1"

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt
import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
import hydra

# import wandb
from omegaconf import DictConfig


@hydra.main(config_path="conf")
def main(cfg: DictConfig):
    # wandb.init(project="sac_siren_brax", config=cfg)
    env = envs.get_environment(env_name=cfg.env_name, backend=cfg.backend)
    if cfg.env_name == "hopper" or cfg.env_name == "walker2d":
        trainer = sac.train
    elif (
        cfg.env_name == "pusher"
        or cfg.env_name == "reacher"
        or cfg.env_name == "humanoidstandup"
        or cfg.env_name == "humanoid"
        or cfg.env_name == "halfcheetah"
        or cfg.env_name == "ant"
        or cfg.env_name == "inverted_double_pendulum"
        or cfg.env_name == "inverted_pendulum"
    ):
        trainer = ppo.train
    else:
        raise ValueError("Invalid environment name")
    train_fn = functools.partial(
        trainer,
        num_timesteps=cfg.num_timesteps,
        num_evals=cfg.num_evals,
        reward_scaling=cfg.reward_scaling,
        episode_length=cfg.episode_length,
        normalize_observations=cfg.normalize_observations,
        action_repeat=cfg.action_repeat,
        unroll_length=cfg.unroll_length,
        num_minibatches=cfg.num_minibatches,
        num_updates_per_batch=cfg.num_updates_per_batch,
        discounting=cfg.discounting,
        learning_rate=cfg.learning_rate,
        entropy_cost=cfg.entropy_cost,
        num_envs=cfg.num_envs,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    xdata, ydata = [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"])
        plt.xlim([0, train_fn.keywords["num_timesteps"]])
        plt.ylim([cfg.min_y, cfg.max_y])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.plot(xdata, ydata)
        plt.show()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    model.save_params("/tmp/params", params)
    params = model.load_params("/tmp/params")
    inference_fn = make_inference_fn(params)

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
        state = jit_env_step(state, act)

    html_str = html.render(env.sys.replace(dt=env.dt), rollout)
    # wandb.log({"trained_agent_html": wandb.Html(html_str)})


if __name__ == "__main__":
    main()
