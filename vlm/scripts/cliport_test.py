import argparse
from distutils.util import strtobool
from pathlib import Path
import os
import random
import cv2
from cliport.agent import TwoStreamClipLingUNetLatTransporterAgent, TransporterLangAgent, ImgLangAgent_6Dof, DepthLangAgent_6Dof, BlindLangAgent_6Dof,\
    TwoStreamClipLingUNetLatTransporterAgent_IGNORE
from amsolver.environment import Environment
from amsolver.action_modes import ArmActionMode, ActionMode
from amsolver.observation_config import ObservationConfig
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode
from amsolver.utils import get_stored_demos
from amsolver.backend.utils import task_file_to_task_class
from num2words import num2words
# from pyvirtualdisplay import Display
# disp = Display().start()
class Recorder(object):
    def __init__(self) -> None:
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        self.cam = VisionSensor.create([640, 360])
        self.cam.set_pose(cam_placeholder.get_pose())
        self.cam.set_parent(cam_placeholder)
        self._snaps = []
        self._fps=30

    def take_snap(self):
        self._snaps.append(
            (self.cam.capture_rgb() * 255.).astype(np.uint8))
    
    def save(self, path):
        print('Converting to video ...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'MJPG'), self._fps,
                tuple(self.cam.get_resolution()))
        for image in self._snaps:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()
        self._snaps = []

class ReplayAgent(object):

     def act(self, step_list, step_id, obs, lang, use_gt_xy=False,use_gt_z=False, use_gt_theta=False, use_gt_roll_pitch=False):
        current_waypoint,_, attention_id, gripper_control, waypoint_type, related_rotation, gt_pose  = step_list[step_id]
        action = np.zeros(8)
        action[:7] = gt_pose
        action[7] = gripper_control
        return action, waypoint_type

class CliportAgent(object):
    def __init__(self, model_name, device_id=0, z_roll_pitch=True, checkpoint=None) -> None:
        cfg = {
            'train':{
                'attn_stream_fusion_type': 'add',
                'trans_stream_fusion_type': 'conv',
                'lang_fusion_type': 'mult',
                'n_rotations':36,
                'batchnorm':False
            }
        }
        device = torch.device(device_id)
        if model_name=="cliport_6dof":
            self.agent = TwoStreamClipLingUNetLatTransporterAgent(name='agent', device=device, cfg=cfg, z_roll_pitch=z_roll_pitch).to(device)
        elif model_name == 'cliport_joint':
            self.agent = TwoStreamClipLingUNetLatTransporterAgent_IGNORE(name="cliport_joint",device=device, cfg=cfg, z_roll_pitch=True).to(device)
        elif model_name == "transporter_6dof":
            self.agent = TransporterLangAgent(name='agent',device=device, cfg=cfg).to(device)
        elif model_name == "imglang_6dof":
            self.agent = ImgLangAgent_6Dof(name='agent',device=device, cfg=cfg).to(device)
        elif model_name == 'depthlang_6dof':
            self.agent = DepthLangAgent_6Dof(name='agent',device=device, cfg=cfg).to(device)
        elif model_name == 'blindlang_6dof':
            self.agent = BlindLangAgent_6Dof(name='agent',device=device, cfg=cfg).to(device)
        self.model_name = model_name
        if checkpoint is not None:
            state_dict = torch.load(checkpoint,device)
            self.agent.load_state_dict(state_dict['state_dict'])
        self.agent.eval()

    @staticmethod
    def generate_action_list(waypoints_info, args):
        all_waypoints = []
        i=0
        while True:
            waypoint_name = f"waypoint{i}"
            i+=1
            if waypoint_name in waypoints_info:
                all_waypoints.append(waypoint_name)
            else:
                break
        step_list, point_list = [], []
        attention_id = waypoints_info["waypoint0"]["target_obj"]
        gripper_control = 1
        for i, wp in enumerate(all_waypoints):
            waypoint_info = waypoints_info[wp]
            waypoint_type = waypoint_info['waypoint_type']
            if "pre" in waypoint_type:
                focus_wp = all_waypoints[i+1]
            elif "post" in waypoint_type:
                focus_wp = all_waypoints[i-1]
            else:
                focus_wp = wp
            if focus_wp not in point_list:
                focus_waypoint_info = waypoints_info[focus_wp]
                if "grasp" in waypoint_type:
                    attention_id = waypoint_info["target_obj"]
                    related_rotation = False
                else:
                    related_rotation = args.relative
                if focus_waypoint_info["gripper_control"] is not None:
                    gripper_control = focus_waypoint_info["gripper_control"][1]
                gt_pose = focus_waypoint_info['pose'][0]
                point_list.append(focus_wp)
                # if "grasp" in waypoint_type:
                #     continue
                step_list.append([focus_wp, focus_waypoint_info['low_level_descriptions'], attention_id, gripper_control, focus_waypoint_info["waypoint_type"], related_rotation, gt_pose])
        return step_list
    
    def act(self, step_list, step_id, obs, lang, use_gt_xy=False,use_gt_z=False, use_gt_theta=False, use_gt_roll_pitch=False):
        current_waypoint,_, attention_id, gripper_control, waypoint_type, related_rotation, gt_pose  = step_list[step_id]
        if self.model_name =="transporter_6dof":
            lang = f"Step {num2words(step_id)}."
        with torch.no_grad():
            inp_img, lang_goal, p0, output_dict = self.agent.act(obs, [lang], bounds = np.array([[-0.05,0.67],[-0.45, 0.45], [0.7, 1.2]]), pixel_size=5.625e-3/2)
        action = np.zeros(8)
        action[:7] = gt_pose
        if not use_gt_xy:
            action[:2] = output_dict['place_xy']
        if not use_gt_z:
            action[2] = output_dict['place_z']
        action[7] = gripper_control
        if related_rotation:
            prev_pose = R.from_quat(self.prev_pose[3:])
            current_pose = R.from_quat(gt_pose[3:])
            rotation = (prev_pose.inv()*current_pose).as_euler('zyx')
            # rotation[rotation<0]+=2*np.pi
            if not use_gt_theta:
                rotation[0] = output_dict['place_theta']
            if not use_gt_roll_pitch:
                rotation[1] = output_dict['pitch']
                rotation[2] = output_dict['roll']
            related_rot = R.from_euler("zyx",rotation)
            action[3:7] = (prev_pose*related_rot).as_quat()
        else:
            rotation = R.from_quat(gt_pose[3:]).as_euler('zyx')
            # rotation[rotation<0]+=2*np.pi
            if not use_gt_theta:
                rotation[0] = output_dict['place_theta']
            if not use_gt_roll_pitch:
                rotation[1] = output_dict['pitch']
                rotation[2] = output_dict['roll']
            action[3:7] = R.from_euler("zyx",rotation).as_quat()
        self.prev_pose = action[:7]
        return action, waypoint_type


def load_test_config(data_folder: Path, task_name):
    episode_list = []
    for path in data_folder.rglob('configs*'):
        t_name = path.parents[3].name
        if t_name == task_name:
            episode_list.append(path.parent)
    return episode_list

def set_seed(seed, torch=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if torch:
        import torch
        torch.manual_seed(seed)

def add_argments():
    parser = argparse.ArgumentParser(description='')
    #dataset
    parser.add_argument('--data_folder', type=str, default="/data1/zhengkz/rlbench_data/test/seen")
    parser.add_argument('--checkpoints_folder', type=str, default="/data1/zhengkz/new_weights")
    parser.add_argument('--model_name', type=str, default="cliport_6dof")
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--replay', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--relative', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--renew_obs', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--add_low_lang', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--ignore_collision', type=lambda x:bool(strtobool(x)), default=False)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = add_argments()
    set_seed(0)
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    img_size=(360,360)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
    obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
    obs_config.overhead_camera.render_mode = RenderMode.OPENGL
    obs_config.wrist_camera.render_mode = RenderMode.OPENGL
    obs_config.front_camera.render_mode = RenderMode.OPENGL

    if args.ignore_collision:
        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
    else:
        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME_WITH_COLLISION_CHECK)
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    # recorder = Recorder()
    recorder = None
    need_test_numbers = 100
    replay_test = args.replay
    
    renew_obs = args.renew_obs
    need_post_grap = True
    need_pre_move = False
    if args.task == 'drop':
        task_files = ['drop_pen_color', 'drop_pen_relative', 'drop_pen_size']
    elif args.task == 'pick':
        task_files = ['pick_cube_shape', 'pick_cube_relative', 'pick_cube_color', 'pick_cube_size']
    elif args.task == 'stack':
        task_files = ['stack_cubes_color', 'stack_cubes_relative', 'stack_cubes_shape', 'stack_cubes_size']
    elif args.task == 'place':
        need_pre_move = True
        task_files = ['place_into_shape_sorter_color', 'place_into_shape_sorter_relative', 'place_into_shape_sorter_shape']
    elif args.task == 'wipe':
        task_files = ['wipe_table_color', 'wipe_table_relative', 'wipe_table_size', 'wipe_table_direction']
    elif args.task == 'pour':
        task_files = ['pour_demo_color', 'pour_demo_relative', 'pour_demo_size']
    elif args.task == 'drawer':
        need_post_grap=False
        task_files = ['open_drawer', 'open_drawer_cabinet']
    elif args.task == 'door':
        need_post_grap=False
        task_files = ['open_door']
    train_tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in task_files]
    data_folder = Path(args.data_folder)
    if not replay_test:
        checkpoint = args.checkpoints_folder + f"/conv_checkpoint_{args.model_name}_{args.task}"
        if args.relative:
            checkpoint += '_relative'
        if args.renew_obs:
            checkpoint += '_renew'
        if args.add_low_lang:
            checkpoint += '_low'
        checkpoint += '_best.pth'
        # checkpoint = "/data1/zhengkz/new_weights/conv_checkpoint_cliport_6dof_stack_best_norenew_low.pth"
        agent = CliportAgent(args.model_name, device_id=args.gpu,z_roll_pitch=True, checkpoint=checkpoint)
    else:
        agent = ReplayAgent()
    for i, task_to_train in enumerate(train_tasks):
        e_path = load_test_config(data_folder, task_files[i])
        success_times = 0
        grasp_success_times = 0
        all_time = 0
        task = env.get_task(task_to_train)
        for num, e in enumerate(e_path):
            if num >= need_test_numbers:
                break
            task_base = str(e/"task_base.ttm")
            waypoint_sets = str(e/"waypoint_sets.ttm")
            config = str(e/"configs.pkl")
            descriptions, obs = task.load_config(task_base, waypoint_sets, config)
            waypoints_info = {name: obj for name, obj in obs.object_informations.items() if "waypoint" in name}
            all_time+=1
            high_descriptions = descriptions[0]
            if high_descriptions[-1]!=".":
                high_descriptions+="."
            print(high_descriptions)
            target_grasp_obj_name = None
            try:
                grasp_pose = waypoints_info['waypoint1']['pose'][0]
                target_name = None
                distance = np.inf
                for g_obj in task._task.get_graspable_objects():
                    obj_name = g_obj.get_name()
                    obj_pos = g_obj.get_position()
                    c_distance = np.linalg.norm(obj_pos-grasp_pose[:3])
                    if c_distance < distance:
                        target_name = obj_name
                        distance = c_distance
                if distance < 0.2:
                    target_grasp_obj_name = target_name
            except:
                print("need re-generate.")
            step_list = CliportAgent.generate_action_list(waypoints_info, args)
            action_list = []
            collision_checking_list = []
            """
            demos = get_stored_demos(1, False, "../vlmbench/train/", 1, 
                                                'wipe_table_color', obs_config, 'episode0', False)
            lang = demos[0].high_level_instructions[0]+f" Step {num2words(i)}."
            agent.act(step_list, i, demos[0]._observations[0], lang, use_gt_z= True, use_gt_theta= False)
            """
            for i, sub_step in enumerate(step_list):
                lang = high_descriptions+f" Step {num2words(i)}."
                if args.add_low_lang:
                    lang += sub_step[1]
                action, action_type = agent.act(step_list, i, obs, lang, use_gt_xy=False, use_gt_z= False, use_gt_theta= False, use_gt_roll_pitch=False)
                if "grasp" in action_type:
                    pre_action = action.copy()
                    pose = R.from_quat(action[3:7]).as_matrix()
                    pre_action[:3] -= 0.08*pose[:, 2]
                    pre_action[7] = 1
                    action_list+=[pre_action, action]
                    collision_checking_list += [True, False]
                    if need_post_grap:
                        post_action = action.copy()
                        post_action[2] = post_action[2] + 0.08
                        action_list += [post_action]
                        collision_checking_list+=[False]
                else:
                    if need_pre_move:
                        pre_action = action.copy()
                        pose = R.from_quat(action[3:7]).as_matrix()
                        pre_action[:3] -= 0.08*pose[:, 2]
                        pre_action[7] = 0
                        action_list+=[pre_action]
                        collision_checking_list += [None]
                    action_list+= [action]
                    collision_checking_list += [None]
                if renew_obs:
                    try:
                        for action, collision_checking in zip(action_list,collision_checking_list):
                            obs, reward, terminate = task.step(action, collision_checking, recorder = recorder, need_grasp_obj = target_grasp_obj_name)
                        action_list = []
                        collision_checking_list = []
                        if reward == 1:
                            success_times+=1
                            break
                        elif reward == 0.5:
                            grasp_success_times += 1
                    except Exception as e:
                        print(e)
                        break
            if not renew_obs and len(action_list):
                # pre_grasp = waypoints_info['waypoint0']['pose'][0]
                # grasp = waypoints_info['waypoint1']['pose'][0]
                # post_grasp = waypoints_info['waypoint2']['pose'][0]
                # action0 = np.zeros(8)
                # action1 = np.zeros(8)
                # action2 = np.zeros(8)
                # action0[:7] = pre_grasp
                # action0[7] = 1
                # action1[:7] = grasp
                # action2[:7] = post_grasp
                # action_list = [action0, action1, action2] + action_list
                # collision_checking_list = [True, False, False] + collision_checking_list
                try:
                    for action, collision_checking in zip(action_list,collision_checking_list):
                        obs, reward, terminate = task.step(action, collision_checking, recorder = recorder, use_auto_move=True, need_grasp_obj = target_grasp_obj_name)
                        if reward == 1:
                            success_times+=1
                            break
                        elif reward == 0.5:
                            grasp_success_times += 1
                except Exception as e:
                    print(e)
            if recorder is not None:
                recorder.save(f"./records/error1_{task.get_name()}.avi")
            print(f"{task.get_name()}: success {success_times} times in {all_time} steps!")
            print(f"{task.get_name()}: grasp success {grasp_success_times} times in {all_time} steps!")
    env.shutdown()