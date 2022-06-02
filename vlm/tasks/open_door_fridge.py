from typing import List
import numpy as np
import os
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.unit_tasks import T0_ObtainControl, T2_MoveObjectConstraints, TargetSpace, VLM_Object
from amsolver.backend.conditions import JointCondition
from amsolver.const import colors
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.joint import Joint

door_states = {
    "open":["Fully close", "Slightly close"],
    "close":["Fully open", "Slightly open"]
}
class OpenDoorFridge(Task):

    def init_task(self) -> None:
        model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        model_path = self.import_model(model_dir)
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.boundary_root_ori = Shape("boundary_root").get_orientation()
        self.attach_point = Dummy.create()
        self.attach_point.set_name("door_attach_point")
        self.attach_point.set_parent(Shape("boundary_root"))
        self.door = VLM_Object(self.pyrep, model_path, 0)
        root_pose = Shape("boundary_root").get_pose()
        root_pose[:2] = self.door.get_position()[:2]
        self.attach_point.set_pose(root_pose)
        self.door.set_parent(self.attach_point)
        self.door_joint = self.door.constraints[0]
        self.door_joint_name = self.door_joint.get_name()
        self.pyrep.step()
        self.temporary_waypoints = []
        self.task_base = self.get_base()
        if not hasattr(self, "_ignore_collisions"):
            self._ignore_collisions = False

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        goal_angel, goal_description = self.door_setting(index)
        try_times = 0
        while len(self.temporary_waypoints)==0 and try_times<100:
            self.sample_method()
            init_states = self.get_state()[0]
        
            self.door.manipulated_part.descriptions = f"the handle of the {self.door.manipulated_part.property['shape']}"
            door_target_description = f"the {self.door.manipulated_part.property['shape']}"
            if "Slightly" in goal_description:
                door_target_description += " slightly"
            door_target = TargetSpace(self.door_joint, None,
                                                np.deg2rad(goal_angel), np.deg2rad(goal_angel), door_target_description, self.door.manipulated_part.visual)
            door_target.set_target(self.door.parts[1])
            door_task = T2_MoveObjectConstraints(self.robot, self.pyrep, door_target, self.task_base, fail_times=2)
            post_grasp_task = door_task
            grasp_task = T0_ObtainControl(self.robot, self.pyrep, self.door.manipulated_part, self.task_base, try_times=100,
                    need_post_grasp=False, grasp_sort_key="horizontal", next_task_fuc=post_grasp_task.get_path_with_constraints)
            if try_times>50:
                waypoints = grasp_task.get_path(try_ik_sampling=True, ignore_collisions=self._ignore_collisions)
            else:
                waypoints = grasp_task.get_path(try_ik_sampling=False, ignore_collisions=self._ignore_collisions)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            try_times += 1
            self.robot.reset()
            self.pyrep.set_configuration_tree(init_states)
            self.pyrep.step()
            for i,waypoint in enumerate(self.temporary_waypoints):
                waypoint.set_name('waypoint{}'.format(i))
        return [f"{goal_description} the {self.door.manipulated_part.property['shape']}."]

    def import_model(self, model_dir):
        model_path = model_dir+"fridge/fridge1/fridge1.ttm"
        return model_path

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 4

    def is_static_workspace(self) -> bool:
        return True
    
    def sample_method(self):
        Shape("boundary_root").set_orientation(self.boundary_root_ori)
        self.spawn_space.clear()
        self.spawn_space.sample(Shape("boundary_root"), 
            min_rotation=[0, 0, -3.14 / 8.], max_rotation=[0, 0, 3.14 / 8.])
        # SpawnBoundary([Shape('boundary_root')]).sample(self.attach_point, 
        #         min_rotation=[0, 0, 0], max_rotation=[0, 0, 0])

    def door_setting(self, index):
        if index in [0,1]:
            init_state = "open"
            self.door_joint.set_joint_position(np.deg2rad(30), True)
        else:
            init_state = "close"
            self.door_joint.set_joint_position(np.deg2rad(0), True)
        sub_index = index%2
        goal_state = door_states[init_state][sub_index]
        if "Slightly" in goal_state:
            detect_angle = 10
            goal_angel = 15
            detect_bound = 20
        else:
            detect_angle = 25
            detect_bound = 35
            if "open" in goal_state:
                goal_angel = 30
            else:
                goal_angel = 0
        self.register_success_conditions([JointCondition(self.door_joint, np.deg2rad(detect_angle), np.deg2rad(detect_bound))])
        return goal_angel, goal_state

    def load(self, ttms_folder=None):
        if Shape.exists('open_door'):
            return Dummy('open_door')
        ttm_file = os.path.join(ttms_folder, 'open_door.ttm')
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
    
    def get_base(self) -> Dummy:
        self._base_object = Dummy('open_door')
        return self._base_object