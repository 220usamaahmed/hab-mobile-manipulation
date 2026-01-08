import argparse
import json
import os
import os.path as osp
import re

import magnum as mn
import numpy as np
import torch
from habitat import Config, logger
from habitat_baselines.utils.common import batch_obs, generate_video

import mobile_manipulation.methods.skills
from habitat_extensions.tasks.rearrange import RearrangeRLEnv
from habitat_extensions.tasks.rearrange.play import get_action_from_key
from habitat_extensions.utils.viewer import OpenCVViewer
from habitat_extensions.utils.visualizations.utils import put_info_on_image
from mobile_manipulation.config import get_config
from mobile_manipulation.methods.skill import CompositeSkill
from mobile_manipulation.utils.common import (
    extract_scalars_from_info,
    get_git_commit_id,
    get_run_name,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1
from habitat.tasks.utils import cartesian_to_polar

import sys


import pickle



def preprocess_config(config_path: str, config: Config):
    config.defrost()

    fileName = osp.splitext(osp.basename(config_path))[0]
    runName = get_run_name()
    substitutes = dict(fileName=fileName, runName=runName)

    config.PREFIX = config.PREFIX.format(**substitutes)
    config.BASE_RUN_DIR = config.BASE_RUN_DIR.format(**substitutes)

    for key in ["LOG_FILE", "VIDEO_DIR"]:
        config[key] = config[key].format(
            prefix=config.PREFIX, baseRunDir=config.BASE_RUN_DIR, **substitutes
        )


def update_ckpt_path(config: Config, seed: int):
    config.defrost()
    for k in config:
        if k == "CKPT_PATH":
            ckpt_path = config[k]
            new_ckpt_path = re.sub(r"seed=[0-9]+", f"seed={seed}", ckpt_path)
            print(f"Update {ckpt_path} to {new_ckpt_path}")
            config[k] = new_ckpt_path
        elif isinstance(config[k], Config):
            update_ckpt_path(config[k], seed)
    config.freeze()


def update_sensor_resolution(config: Config, height, width):
    config.defrost()
    sensor_names = [
        "THIRD_RGB_SENSOR",
        "RGB_SENSOR",
        "DEPTH_SENSOR",
        "SEMANTIC_SENSOR",
    ]
    for name in sensor_names:
        sensor_cfg = config.TASK_CONFIG.SIMULATOR[name]
        sensor_cfg.HEIGHT = height
        sensor_cfg.WIDTH = width
        print(f"Update {name} resolution")
    config.freeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="config_path", type=str, required=True)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    # Episodes
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="whether to shuffle test episodes",
    )
    parser.add_argument(
        "--num-episodes", type=int, help="number of episodes to evaluate"
    )
    parser.add_argument(
        "--episode-ids", type=str, help="episodes ids to evaluate"
    )

    # Save
    parser.add_argument("--save-video", choices=["all", "failure"])
    parser.add_argument("--save-log", action="store_true")

    # Viewer
    parser.add_argument(
        "--viewer", action="store_true", help="enable OpenCV viewer"
    )
    parser.add_argument("--viewer-delay", type=int, default=10)
    parser.add_argument(
        "--play", action="store_true", help="enable input control"
    )

    # Policy
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-seed", type=int)

    # Rendering
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--render-info", action="store_true")
    parser.add_argument(
        "--no-rgb", action="store_true", help="disable rgb observations"
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="use high resolution for visualization",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------------- #
    # Configure
    # ---------------------------------------------------------------------------- #
    config = get_config(args.config_path, opts=args.opts)
    preprocess_config(args.config_path, config)
    torch.set_num_threads(1)

    config.defrost()
    if args.split is not None:
        config.TASK_CONFIG.DATASET.SPLIT = args.split
    if not args.shuffle:
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = False
    if args.no_rgb:
        sensors = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = [
            x for x in sensors if "RGB" not in x
        ]
    config.freeze()

    if args.train_seed is not None:
        update_ckpt_path(config, seed=args.train_seed)

    if args.high_res:
        update_sensor_resolution(config, height=720, width=1080)

    if args.save_log:
        if config.LOG_FILE:
            log_dir = os.path.dirname(config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(config.LOG_FILE)
        logger.info(config)
        logger.info("commit id: {}".format(get_git_commit_id()))

    # For reproducibility, just skip other episodes
    if args.episode_ids is not None:
        eval_episode_ids = eval(args.episode_ids)
        eval_episode_ids = [str(x) for x in eval_episode_ids]
    else:
        eval_episode_ids = None

    # ---------------------------------------------------------------------------- #
    # Initialize env
    # ---------------------------------------------------------------------------- #
    env = RearrangeRLEnv(config)
    env = HabitatActionWrapperV1(env)
    env.seed(config.TASK_CONFIG.SEED)
    print("obs space", env.observation_space)
    print("action space", env.action_space)

    # -------------------------------------------------------------------------- #
    # Initialize policy
    # -------------------------------------------------------------------------- #
    policy = CompositeSkill(config.SOLUTION, env)
    policy.to(args.device)

    # -------------------------------------------------------------------------- #
    # Main
    # -------------------------------------------------------------------------- #
    num_episodes = env.number_of_episodes
    # num_episodes = len(env.habitat_env.episode_iterator.episodes)
    if args.num_episodes is not None:
        num_episodes = args.num_episodes

    done, info = True, {}
    all_episode_stats = []
    episode_reward = 0
    failure_episodes = []

    if args.save_video is not None:
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
    rgb_frames = []

    if args.viewer:
        viewer = OpenCVViewer(config.TASK_CONFIG.TASK.TYPE)






    robot_head_depth_temp=np.array([])
    robot_arm_depth_temp=np.array([])
    rob_qpos_temp=np.array([])
    rel_resting_pos_temp=np.array([])
    rel_pick_pos_ee_temp=np.array([])
    rel_place_pos_ee_temp=np.array([])
    rel_pick_pos_base_temp=np.array([])
    rel_place_pos_base_temp=np.array([])
    rel_pick_pos_base_polar_temp=np.array([])
    rel_place_pos_base_polar_temp=np.array([])
    is_holding_temp=np.array([])
    action_to_save_temp=np.array([])

    dataset_dict={}
    episode_ids=[]


    number_of_episodes=0
    num_suc_episodes=0
    total_steps=0
    saves=0
    for i_ep in range(num_episodes):
        ob = env.reset()
        initial_robot_pos = env.env._env._sim.robot.base_pos
        policy.reset(ob)

        episode_reward = 0.0
        info = {}
        rgb_frames = []
        episode_id = env.current_episode.episode_id
        scene_id = env.current_episode.scene_id

        
        #print("current episode == " , env.current_episode.target_receptacles[0][1])


        # Skip episode and keep reproducibility
        if eval_episode_ids is not None and episode_id not in eval_episode_ids:
            print("Skip episode", episode_id)
            continue




        number_of_steps=0
        while True:
            step_action = policy.act(ob)
            if step_action is None:
                print("Terminate the episode given none action")
                break

            # -------------------------------------------------------------------------- #
            # Visualization
            # -------------------------------------------------------------------------- #
            if args.viewer or args.save_video:
                # Add additional info
                info["values"] = step_action.get("values")
                info["value"] = step_action.get("value")
                info["success_probs"] = step_action.get("success_probs")

                metrics = extract_scalars_from_info(info)
                if args.render_mode == "human":
                    frame = env.render(
                        "human",
                        info=metrics,
                        overlay_info=False,
                        show_info=args.render_info,
                    )
                else:
                    frame = env.render(args.render_mode)
                    if args.render_info:
                        frame = put_info_on_image(
                            frame, info=metrics, overlay=False
                        )
                rgb_frames.append(frame)

            if args.viewer:
                key = viewer.imshow(
                    frame[..., :3], delay=0 if args.play else args.viewer_delay
                )

            if args.play:
                play_action = get_action_from_key(key, "BaseArmGripperAction")
                if play_action is not None:
                    step_action = play_action
            # -------------------------------------------------------------------------- #




            #######################################################
            #############   Data collection code  #################
            ## This data is collected before action execution ##

            robot_base_pos= env.env._env._sim.robot.base_pos
            robot_base_orientation= env.env._env._sim.robot.base_ori
            robot_qpos= env.env._env._sim.robot.arm_joint_pos 	
           # robot_ee_T= env.env._env._sim.robot.ee_T 	
            robot_ee_pos=env.env._env._sim.robot.gripper_T.translation
        
            gripper_is_grasped= env.env._env._sim.gripper.is_grasped 	
            if gripper_is_grasped:
            	grasped=1
            else:
            	grasped=-1


            pick_goal=   env.env._env._task.pick_goal    
            place_goal=   env.env._env._task.place_goal 
            resting_pos=env.env._env._task.resting_position   


            ## rgb and depth images has the format (W,H,C) where C is the number of channels
            ## You will need to use permute function to change it to (C,W,H) to be compatible with pytorch
            robot_head_rgb = ob['robot_head_rgb']
            robot_arm_rgb = ob['robot_arm_rgb']
            robot_head_depth = ob['robot_head_depth']
            robot_arm_depth = ob['robot_arm_depth']

            robot_transform = env.env._env._sim.robot.base_T
            robot_ee_transform=env.env._env._sim.robot.gripper_T
            local_ee_pos_relative_to_base=robot_transform.inverted().transform_point(robot_ee_pos)
          #  abs_ee_pos = env.env._env._sim.robot.ee_transform.translation
            relative_pick_pos_base = robot_transform.inverted().transform_point(pick_goal)
            relative_pick_pos_base_polar=cartesian_to_polar(relative_pick_pos_base[0], relative_pick_pos_base[2])
            relative_pick_pos_ee=robot_ee_transform.inverted().transform_point(pick_goal)

            relative_place_pos_base = robot_transform.inverted().transform_point(place_goal)
            relative_place_pos_base_polar=cartesian_to_polar(relative_place_pos_base[0], relative_place_pos_base[2])
            relative_place_pos_ee = robot_ee_transform.inverted().transform_point(place_goal)


            current_task_name= policy.current_skill_name
            if step_action['action']=='EmptyAction' and current_task_name!="ResetArm":
                action_to_save = np.zeros((1,10))
            elif step_action['action']=='EmptyAction' and current_task_name=="ResetArm":
                action_to_save = np.zeros((1,10))
                if gripper_is_grasped:
                    action_to_save[0][9]=1
                else:
                    action_to_save[0][9]=-1
                action_to_save[0][2:9]=policy.current_skill.incremental_qpos_act  
                action_to_save[0][0:2]=np.array([0.0,0.0])

            elif step_action['action']=='BaseArmGripperAction2':
                ## for the BaseArmGripperAction2 the velocity of the base is multiplied by 1.5 (not 3 as in baseDiscVelAction)
                action_to_save=np.array([step_action['action_args']])
                action_to_save[0,0:2]*=1.5
               # print("actions to save shape for pick task == ", action_to_save.shape)

            elif step_action['action']=='BaseDiscVelAction':
                ## baseDiscVelAction has 20 discrete actions, and the policy is categorical distribution over these 20 actions
                ## then the discrete action is mapped to a continuous velocity command between -1 and 1
                ## then the continuous velocity command is multiplied by 3 to get the actual velocity command sent to the robot
                discrete_nav_action=step_action['action_args']                                  
                possible_velocities = np.array(
                [
                    [lin_vel, ang_vel]
                    for lin_vel in np.linspace(-0.5, 1.0, 4)
                    for ang_vel in np.linspace(-1.0, 1.0, 5)
                ]
                )
                current_velocity=possible_velocities[discrete_nav_action]
                if gripper_is_grasped:
                    action_to_save=np.array([[current_velocity[0]*3 , current_velocity[1]*3 , 0 , 0, 0 ,0 , 0 ,0 ,0 , 1 ]])
                else:
                    action_to_save=np.array([[current_velocity[0]*3 , current_velocity[1]*3  , 0 , 0, 0 ,0 , 0 ,0 ,0 , -1 ]]) ## the gripper action should be -1
               # print("actions to save shape for nav task == ", action_to_save.shape)
            else:
                print("Unknown action type")
                input()



            if number_of_steps==0:
                robot_head_depth_temp=[robot_head_depth]
                robot_arm_depth_temp=[robot_arm_depth]
                rob_qpos_temp=[robot_qpos]
                rel_resting_pos_temp=[local_ee_pos_relative_to_base-resting_pos]
                rel_pick_pos_ee_temp=[relative_pick_pos_ee]
                rel_place_pos_ee_temp=[relative_place_pos_ee]
                rel_pick_pos_base_polar_temp=[relative_pick_pos_base_polar]
                rel_place_pos_base_polar_temp=[relative_place_pos_base_polar]
                rel_pick_pos_base_temp=[relative_pick_pos_base]
                rel_place_pos_base_temp=[relative_place_pos_base]
                is_holding_temp=[np.array([gripper_is_grasped])]
                action_to_save_temp=[action_to_save[0]]


            else:
                robot_head_depth_temp=np.append( robot_head_depth_temp,[robot_head_depth],axis=0)
                robot_arm_depth_temp=np.append( robot_arm_depth_temp,[robot_arm_depth],axis=0)
                rob_qpos_temp=np.append( rob_qpos_temp,[robot_qpos],axis=0)
                rel_resting_pos_temp=np.append( rel_resting_pos_temp,[local_ee_pos_relative_to_base-resting_pos],axis=0)
                rel_pick_pos_ee_temp=np.append( rel_pick_pos_ee_temp,[relative_pick_pos_ee],axis=0)
                rel_place_pos_ee_temp=np.append( rel_place_pos_ee_temp,[relative_place_pos_ee],axis=0)
                rel_pick_pos_base_polar_temp=np.append( rel_pick_pos_base_polar_temp,[relative_pick_pos_base_polar],axis=0)
                rel_place_pos_base_polar_temp=np.append( rel_place_pos_base_polar_temp,[relative_place_pos_base_polar],axis=0)
                rel_pick_pos_base_temp=np.append( rel_pick_pos_base_temp,[relative_pick_pos_base],axis=0)
                rel_place_pos_base_temp=np.append( rel_place_pos_base_temp,[relative_place_pos_base],axis=0)
                is_holding_temp=np.append( is_holding_temp,[np.array([gripper_is_grasped])],axis=0)
                action_to_save_temp=np.append( action_to_save_temp,[action_to_save[0]],axis=0)
            




            ob, reward, done, info = env.step(step_action)
            episode_reward += reward


            number_of_steps+=1

            if args.viewer and key == "r":
                done = True
            if done:
                break

        # -------------------------------------------------------------------------- #
        # Update stats
        # -------------------------------------------------------------------------- #
        metrics = extract_scalars_from_info(info)
        episode_stats = metrics.copy()
        episode_stats["return"] = episode_reward
        all_episode_stats.append(episode_stats)

        logger.info(
            "Episode {} ({}/{}): {}".format(
                episode_id, i_ep, num_episodes, episode_stats
            )
        )

        success = metrics.get(config.RL.SUCCESS_MEASURE, -1)
        is_failure = success == False
        print("success == " , success)
        if success:

            print("successful trajectory")
            print("Episode ID == " , env.current_episode.episode_id)
            episode_ids.append(env.current_episode.episode_id)
            

            print("robot_head_depth_temp shape == ", robot_head_depth_temp.shape)
            print("robot_arm_depth_temp shape == ", robot_arm_depth_temp.shape)
            print("rob_qpos_temp shape == ", rob_qpos_temp.shape)
            print("rel_resting_pos_temp shape == ", rel_resting_pos_temp.shape)
            print("rel_pick_pos_ee_temp shape == ", rel_pick_pos_ee_temp.shape)
            print("rel_place_pos_ee_temp shape == ", rel_place_pos_ee_temp.shape)
            print("rel_pick_pos_base_polar_temp shape == ", rel_pick_pos_base_polar_temp.shape)
            print("rel_place_pos_base_polar_temp shape == ", rel_place_pos_base_polar_temp.shape)
            print("rel_pick_pos_base_temp shape == ", rel_pick_pos_base_temp.shape)
            print("rel_place_pos_base_temp shape == ", rel_place_pos_base_temp.shape)
            print("is_holding_temp shape == ", is_holding_temp.shape)
            print("action_to_save_temp shape == ", action_to_save_temp.shape)
            

            dataset_dict['{}'.format(num_suc_episodes)]={}
            dataset_dict['{}'.format(num_suc_episodes)]['robot_head_depth']=robot_head_depth_temp
            dataset_dict['{}'.format(num_suc_episodes)]['robot_arm_depth']=robot_arm_depth_temp
            dataset_dict['{}'.format(num_suc_episodes)]['rob_qpos']=rob_qpos_temp
            dataset_dict['{}'.format(num_suc_episodes)]['rel_resting_pos']=rel_resting_pos_temp
            dataset_dict['{}'.format(num_suc_episodes)]['rel_pick_pos_ee']=rel_pick_pos_ee_temp
            dataset_dict['{}'.format(num_suc_episodes)]['rel_place_pos_ee']=rel_place_pos_ee_temp
            dataset_dict['{}'.format(num_suc_episodes)]['rel_pick_pos_base']=rel_pick_pos_base_temp
            dataset_dict['{}'.format(num_suc_episodes)]['rel_place_pos_base']=rel_place_pos_base_temp
            dataset_dict['{}'.format(num_suc_episodes)]['rel_pick_pos_base_polar']=rel_pick_pos_base_polar_temp
            dataset_dict['{}'.format(num_suc_episodes)]['rel_place_pos_base_polar']=rel_place_pos_base_polar_temp
            dataset_dict['{}'.format(num_suc_episodes)]['is_holding']=is_holding_temp
            dataset_dict['{}'.format(num_suc_episodes)]['action_to_save']=action_to_save_temp
           
            
                        
            robot_head_depth_temp=np.array([])
            robot_arm_depth_temp=np.array([])
            rob_qpos_temp=np.array([])
            rel_resting_pos_temp=np.array([])
            rob_qpos_temp=np.array([])
            rel_pick_pos_ee_temp=np.array([])
            rel_place_pos_ee_temp=np.array([])
            rel_pick_pos_base_temp=np.array([])
            rel_place_pos_base_temp=np.array([])
            rel_pick_pos_base_polar_temp=np.array([])
            rel_place_pos_base_polar_temp=np.array([])
            is_holding_temp=np.array([])
            action_to_save_temp=np.array([])


            total_steps+=number_of_steps

            num_suc_episodes+=1
            print("total number of episodes == " , number_of_episodes+1)
            print("successful episodes == " , num_suc_episodes )
            print("total number of steps == " , total_steps)
            print("len(all_episode_stats) == " , len(all_episode_stats) )
            if num_suc_episodes%50 == 49:
                saves+=1
            
                with open('/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/part_{}.pkl'.format(saves), 'wb') as f:
                    pickle.dump(dataset_dict, f)
                    dataset_dict={} 





        if args.save_video == "all" or (
            args.save_video == "failure" and is_failure
        ):
            generate_video(
                video_option=["disk"],
                video_dir=config.VIDEO_DIR,
                images=rgb_frames,
                episode_id=episode_id,
                checkpoint_idx=-1,
                metrics={"success": success},
                fps=30,
                tb_writer=None,
            )

        if is_failure:
            failure_episodes.append(episode_id)

        if eval_episode_ids is not None:
            if len(all_episode_stats) >= len(eval_episode_ids):
                print("Completed")
                break

        number_of_episodes+=1

    env.close()

    # logging metrics
    aggregated_stats = {
        k: np.mean([ep_info[k] for ep_info in all_episode_stats])
        for k in all_episode_stats[0].keys()
    }
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.4f}")

    failure_episodes = sorted(failure_episodes)
    failure_episodes_str = ",".join(map(str, failure_episodes))
    logger.info("Failure episodes:\n{}".format(failure_episodes_str))

    if args.save_log:
        json_path = config.LOG_FILE.replace("log.txt", "result.json")
        with open(json_path, "w") as f:
            json.dump(all_episode_stats, f, indent=2)


if __name__ == "__main__":
    main()
