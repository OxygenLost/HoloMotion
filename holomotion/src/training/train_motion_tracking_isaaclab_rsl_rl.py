import argparse
import os
import sys
from datetime import datetime

import torch
from omegaconf import OmegaConf
from isaaclab.app import AppLauncher

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# Ensure bundled rsl_rl package is importable
def _add_rsl_rl_to_path():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    rsl_rl_root = os.path.join(repo_root, "rsl_rl")
    if rsl_rl_root not in sys.path:
        sys.path.insert(0, rsl_rl_root)


_add_rsl_rl_to_path()

from rsl_rl.algorithms import PPO as RslRlPPO  # noqa: E402
from rsl_rl.runners.on_policy_runner import OnPolicyRunner  # noqa: E402


def build_train_cfg(log_dir: str):
    # Matches the provided rsl_rl configuration
    cfg = {
        "seed": 42,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "num_steps_per_env": 24,
        "max_iterations": 30000,
        "empirical_normalization": True,
        "obs_groups": {  # minimal mapping to our env keys
            "policy": ["actor_obs"],
            "critic": ["critic_obs"],
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.005,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "symmetry_cfg": None,
            "rnd_cfg": None,
        },
        "clip_actions": None,
        "save_interval": 500,
        "experiment_name": "g1_flat",
        "run_name": "bydmmc_reproduce",
        "logger": "tensorboard",
        "neptune_project": "isaaclab",
        "wandb_project": "isaaclab",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
    }
    cfg["log_dir"] = log_dir
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Motion Tracking with rsl_rl PPO"
    )
    # Avoid loading YAML; accept nothing for env config and build from defaults
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Logging directory. If not set, a timestamped directory under ./runs will be used.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim headless.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.log_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = os.path.abspath(os.path.join("runs", f"rsl_rl_{ts}"))
    os.makedirs(args.log_dir, exist_ok=True)

    headless = True
    device = "cuda"

    app_launcher_flags = {
        "headless": headless,
        "enable_cameras": not headless,
    }
    _sim_app_launcher = AppLauncher(**app_launcher_flags)
    _sim_app = _sim_app_launcher.app
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
    from holomotion.src.env.motion_tracking_isaaclab import MotionTrackingEnv  # noqa: E402

    # Build a minimal in-code environment config (avoid YAML load)
    # Reuse hydra-managed defaults by reading a known runtime-dumped cfg if present, otherwise compose minimal dict
    default_cfg_path = os.path.abspath(
        os.path.join(os.getcwd(), "isaaclab_env_cfg.yaml")
    )
    if os.path.exists(default_cfg_path):
        try:
            env_cfg = OmegaConf.create(
                {}
            )  # placeholder; env will ignore since it constructs internally
        except Exception:
            env_cfg = OmegaConf.create({})
    else:
        env_cfg = OmegaConf.create({})

    # Create IsaacLab environment and wrap for rsl_rl
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mt_env = MotionTrackingEnv(
        env_cfg, device=device, log_dir=args.log_dir, headless=args.headless
    )
    vec_env = RslRlVecEnvWrapper(mt_env._env)

    # Build training configuration
    train_cfg = build_train_cfg(args.log_dir)

    # Seed
    torch.manual_seed(train_cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg["seed"])

    # Runner and train
    runner = OnPolicyRunner(
        vec_env,
        train_cfg,
        log_dir=args.log_dir,
        device=train_cfg["device"],
    )
    runner.learn(
        num_learning_iterations=train_cfg["max_iterations"],
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
