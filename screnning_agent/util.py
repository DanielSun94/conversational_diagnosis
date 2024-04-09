import gymnasium as gym
import os
from stable_baselines3.common.monitor import Monitor
from typing import Any, Callable, Dict, Optional, Type, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env

load_model_name = '20240124182937_520.pkl'
save_folder = os.path.abspath('/home/disk/sunzhoujian/screen_disease/model_save')


class LinearSchedule:
    def __init__(self, initial_value: float):
        self.initial_value = initial_value

    def __call__(self, progress_remaining: float) -> float:
        if progress_remaining > 0.9:
            lr = (1-progress_remaining) * 10 * self.initial_value
        elif progress_remaining > 0.6:
            lr = (1 - (0.9 - progress_remaining) / 3 * 9) * self.initial_value
        else:
            lr = (1 - (0.6 - progress_remaining) / 6 * 10) * 0.1 * self.initial_value
        return lr

def make_vec_env(
    env_callable: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    monitor_dir: Optional[str] = None,
    seed: Optional[int] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, ]]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default, it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_callable: either the env ID, the env class or a callable returning an env
    :param seed:
    :param n_envs: the number of environments you wish to have in parallel
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            env_kwargs['rank'] = rank
            env_kwargs['total_env'] = n_envs

            env = env_callable(**env_kwargs)
            env = _patch_env(env)

            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)
            return env
        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv
    vec_env = vec_env_cls([make_env(i) for i in range(n_envs)])
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env

