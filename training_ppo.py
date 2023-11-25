import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from collections import OrderedDict
import torch.nn as nn
import argparse
# from dexart.env.create_env import create_env
# from dexart.env.task_setting import TRAIN_CONFIG, IMG_CONFIG, RANDOM_CONFIG
from stable_baselines3.common.torch_layers import PointNetImaginationExtractorGP
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
import torch
from time import time
import numpy as np

from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv
from dexpoint.real_world.task_setting import IMG_CONFIG
from dexpoint.real_world import task_setting


def create_env_fn():
    object_names = ["mustard_bottle"]#"mustard_bottle"]#, "tomato_soup_can", "potted_meat_can"]
    object_name = np.random.choice(object_names)
    rotation_reward_weight = 0  # whether to match the orientation of the goal pose
    use_visual_obs = True
    env_params = dict(object_name=object_name, rotation_reward_weight=rotation_reward_weight,
                        randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=False,
                        no_rgb=True)

    # If a computing device is provided, designate the rendering device.
    # On a multi-GPU machine, this sets the rendering GPU and RL training GPU to be the same,
    # based on "CUDA_VISIBLE_DEVICES".
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    
    environment = AllegroRelocateRLEnv(**env_params)

    # environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

    # Create camera
    environment.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

    # Specify observation
    environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

    # Specify imagination
    environment.setup_imagination_config(task_setting.IMG_CONFIG["relocate_robot_only"])

    
    return environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--freeze', dest='freeze', action='store_true', default=False)
    parser.add_argument('--task_name', type=str, default="mustard_bottle")
    parser.add_argument('--extractor_name', type=str, default="smallpn")
    parser.add_argument('--pretrain_path', type=str, default=None)
    args = parser.parse_args()

    task_name = args.task_name
    extractor_name = args.extractor_name
    seed = args.seed if args.seed >= 0 else random.randint(0, 100000)
    
    pretrain_path = args.pretrain_path
    horizon = 200
    env_iter = args.iter * horizon * args.n
    print(f"freeze: {args.freeze}")

    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")  # train on a list of envs.

    model = PPO("PointCloudPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.n // args.workers) * horizon,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=seed,# policy_kwargs=get_3d_policy_kwargs(extractor_name=extractor_name),
                min_lr=args.lr,
                max_lr=args.lr,
                adaptive_kl=0.02,
                target_kl=0.2,
                )

    if pretrain_path is not None:
        state_dict: OrderedDict = torch.load(pretrain_path)
        model.policy.features_extractor.extractor.load_state_dict(state_dict, strict=False)
        print("load pretrained model: ", pretrain_path)

    rollout = int(model.num_timesteps / (horizon * args.n))

    # after loading or init the model, then freeze it if needed
    if args.freeze:
        model.policy.features_extractor.extractor.eval()
        for param in model.policy.features_extractor.extractor.parameters():
            param.requires_grad = False
        print("freeze model!")

    model.learn(
        total_timesteps=int(env_iter),
        reset_num_timesteps=False,
        iter_start=rollout,
        callback=None
    )