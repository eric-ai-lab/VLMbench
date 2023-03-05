import argparse
from distutils.util import strtobool
from pathlib import Path
import os
from amsolver.environment import Environment
from amsolver.action_modes import ArmActionMode, ActionMode
from amsolver.observation_config import ObservationConfig
import numpy as np
from amsolver.backend.utils import task_file_to_task_class

"""
Data structure of Observation:
{
    'front_rgb': np.array,
    'front_depth': np.array,
    'front_mask': np.array,
    'front_point_cloud': np.array,
    'left_shoulder_rgb': np.array,
    'left_shoulder_depth': np.array,
    'left_shoulder_mask': np.array,
    'left_shoulder_point_cloud': np.array,
    'right_shoulder_rgb': np.array,
    'right_shoulder_depth': np.array,
    'right_shoulder_mask': np.array,
    'right_shoulder_point_cloud': np.array,
    'wrist_rgb': np.array,
    'wrist_depth': np.array,
    'wrist_mask': np.array,
    'wrist_point_cloud': np.array,
    'overhead_rgb': np.array,
    'overhead_depth': np.array,
    'overhead_mask': np.array,
    'overhead_point_cloud': np.array,
    'gripper_joint_positions': np.array,
    'gripper_touch_forces': np.array,
    'gripper_pose': np.array,
    'gripper_matrix': np.array,
    'gripper_open': np.array,
    'joint_positions': np.array,
    'joint_velocities': np.array,
    'joint_forces': np.array,
    'misc': dict, # contains camera extrinsics, camera intrinsics, near/far clipping planes, etc.
    'object_informations': dict, # contains two types of information: object information and waypoint information \
        for each object, the dict includes the convex hull of the object and the visual hull of the object e.g. 'pencil1_0' and 'pencil1_visual0'\
        for each waypoint, the dict includes the pose, type, target object (this waypoint is used for moving/getting close to which object), and low-level description of the waypoint.
}
"""

task_dict = {
    'drop': ['drop_pen_color', 'drop_pen_relative', 'drop_pen_size'],
    'pick': ['pick_cube_shape', 'pick_cube_relative', 'pick_cube_color', 'pick_cube_size'],
    'stack': ['stack_cubes_color', 'stack_cubes_relative', 'stack_cubes_shape', 'stack_cubes_size'],
    'shape_sorter': ['place_into_shape_sorter_color', 'place_into_shape_sorter_relative', 'place_into_shape_sorter_shape'],
    'wipe': ['wipe_table_shape', 'wipe_table_color', 'wipe_table_relative', 'wipe_table_size', 'wipe_table_direction'],
    'pour': ['pour_demo_color', 'pour_demo_relative', 'pour_demo_size'],
    'drawer': ['open_drawer'],
    'door': ['open_door'], # untested
    'door_complex': ['open_door_complex'],
}

class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs, descriptions):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape-1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)
    
def add_argments():
    parser = argparse.ArgumentParser(description='')
    #dataset
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--setd', type=str, default="seen")
    parser.add_argument('--img_size',nargs='+', type=int, default=[360,360])
    parser.add_argument('--task', type=str, default=None, help="select a task type from drop, pick, stack, shape_sorter, wipe, pour, drawer, door_complex")
    parser.add_argument('--use_collect_data', type=lambda x:bool(strtobool(x)), default=True)
    args = parser.parse_args()
    return args

def load_test_config(data_folder: Path, task_name):
    episode_list = []
    for path in data_folder.rglob('configs*'):
        t_name = path.parents[3].name
        if t_name == task_name:
            episode_list.append(path.parent)
    return episode_list

if __name__=="__main__":
    args = add_argments()
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.set_image_size(args.img_size)

    if args.task in task_dict:
        task_files = task_dict[args.task]
    else:
        task_files = [args.task]

    eval_tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in task_files]
    data_folder = Path(os.path.join(args.data_folder, args.setd))

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(action_mode, obs_config=obs_config, headless=False) # set headless=False, if user want to visualize the simulator
    env.launch()

    agent = Agent(env.action_size)
    need_test_numbers = 10
    action_steps = 10
    for i, task_to_use in enumerate(eval_tasks):
        task = env.get_task(task_to_use)

        if args.use_collect_data:
            e_path = load_test_config(data_folder, task_files[i])
            for num, e in enumerate(e_path):
                if num >= need_test_numbers:
                    break
                task_base = str(e/"task_base.ttm")
                waypoint_sets = str(e/"waypoint_sets.ttm")
                config = str(e/"configs.pkl")
                descriptions, obs = task.load_config(task_base, waypoint_sets, config)
                print(descriptions)
                for _ in range(action_steps):
                    action = agent.act(obs, descriptions)
                    obs, reward, terminate = task.step(action)
        else:
            for _ in range(need_test_numbers):
                descriptions, obs = task.reset()
                print(descriptions)
                for _ in range(action_steps):
                    action = agent.act(obs, descriptions)
                    obs, reward, terminate = task.step(action)
    
    env.shutdown()