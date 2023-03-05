import gymnasium as gym
import amsolver.gym

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

env = gym.make('drop_pen_color-vision-v0', render_mode='human')

training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
        descriptions = obs['descriptions']
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    env.render()  # Note: rendering increases step time.

print('Done')
env.close()