import os
import numpy as np
from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task
from amsolver.backend.conditions import JointCondition
from amsolver.backend.unit_tasks import T0_ObtainControl, T2_MoveObjectConstraints, TargetSpace, VLM_Object

drawer_states = {
    "open":["Fully close", "Slightly close"],
    "close":["Fully open", "Slightly open"]
}

class OpenDrawer(Task):

    def init_task(self) -> None:
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        model_path = self.model_dir+"drawer/drawer1/drawer1.ttm"
        self.drawer = VLM_Object(self.pyrep, model_path, 0)
        self._drawer_init_ori = self.drawer.get_orientation()
        self._drawer_init_state = self.drawer.get_configuration_tree()
        self.temporary_waypoints = []
        self.task_base = self.get_base()
        self.drawer.set_parent(self.task_base)
        self.drawer_relative = ["top", "middle", "bottom"]

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        select_index = index // 4
        setting_index = index % 4
        self.manipulate_drawer = self.drawer.manipulated_parts[select_index]
        self.drawer_joint = self.drawer.constraints[select_index]
        drawer_relative_description = self.drawer_relative[select_index]
        try_times = 0
        while len(self.temporary_waypoints)==0 and try_times<200:
            self.sample_method()
            goal_angel, goal_description = self.drawer_setting(setting_index)
            init_states = self.get_state()[0]
        
            self.manipulate_drawer.description = f"the handle of {drawer_relative_description} drawer"
            drawer_target_description = f"the {drawer_relative_description} drawer"
            if "Slightly" in goal_description:
                drawer_target_description += " slightly"
            drawer_target = TargetSpace(self.drawer_joint , None,goal_angel, goal_angel, drawer_target_description, self.manipulate_drawer.visual)
            drawer_target.set_target(self.manipulate_drawer)
            drawer_task = T2_MoveObjectConstraints(self.robot, self.pyrep, drawer_target, self.task_base, fail_times=2)
            grasp_task = T0_ObtainControl(self.robot, self.pyrep, self.manipulate_drawer, self.task_base, try_times=100,
                    need_post_grasp=False, grasp_sort_key="horizontal", next_task_fuc=drawer_task.get_path_with_constraints)
            if try_times>100:
                waypoints = grasp_task.get_path(try_ik_sampling=True, ignore_collisions=False)
            else:
                waypoints = grasp_task.get_path(try_ik_sampling=False, ignore_collisions=False)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            try_times += 1
            self.robot.reset()
            self.pyrep.set_configuration_tree(init_states)
            self.pyrep.step()
            # print(f"Have tried {try_times} times for open drawer.")
            for i,waypoint in enumerate(self.temporary_waypoints):
                waypoint.set_name('waypoint{}'.format(i))
        return [f"{goal_description} the {drawer_relative_description} drawer."]

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 12

    def is_static_workspace(self) -> bool:
        return True

    def sample_method(self):
        self.pyrep.set_configuration_tree(self._drawer_init_state)
        self.drawer.set_orientation(self._drawer_init_ori)
        self.spawn_space.clear()
        # self.drawer.set_model(False)
        self.spawn_space.sample(self.drawer, 
            min_rotation=[0, 0, -3.14 / 4.], max_rotation=[0, 0, 3.14 / 4.])
        # self.drawer.set_model(True)
        self.drawer.set_position(self.drawer.get_position()+np.array([0.1,0,0]))
        self.pyrep.step()

    def drawer_setting(self, index):
        if index in [2,3]:
            init_state = "open"
            self.drawer_joint.set_joint_position(0.18, True)
        else:
            init_state = "close"
            self.drawer_joint.set_joint_position(0, True)
        sub_index = index%2
        goal_state = drawer_states[init_state][sub_index]
        if "Slightly" in goal_state:
            detect_distance = 0.05
            goal_distance = 0.1
            detect_bound = 0.1
        else:
            detect_distance = 0.15
            detect_bound = 0.25
            if "open" in goal_state:
                goal_distance = 0.2
            else:
                goal_distance = 0
        self.register_success_conditions([JointCondition(self.drawer_joint, detect_distance, detect_bound)])
        return goal_distance, goal_state
