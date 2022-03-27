from turtle import shape
from typing import List
import numpy as np
import os
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.task import Task
from amsolver.backend.unit_tasks import T0_ObtainControl, T1_MoveObjectGoal, TargetSpace, VLM_Object
from amsolver.backend.utils import get_relative_position_xy
from amsolver.const import colors
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
        
class PickCube(Task):
    def init_task(self) -> None:
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.target_space0 = TargetSpace(SpawnBoundary([Shape('target_space0')]), ProximitySensor('success0'),
                                        (-3.14,-3.14,-3.14), (3.14,3.14,3.14), None, Shape("small_container0").get_handle())
        self.target_space1 = TargetSpace(SpawnBoundary([Shape('target_space1')]), ProximitySensor('success1'),
                                        (-3.14,-3.14,-3.14), (3.14,3.14,3.14), None, Shape("small_container1").get_handle())
        self.target_spaces = [self.target_space0, self.target_space1]
        self.model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        self.temporary_waypoints = []
        self.object_list = []
        self.taks_base = self.get_base()
        if not hasattr(self, "model_num"):
            self.model_num = 2
        self.import_objects(self.model_num)
    
    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        # ind = np.random.randint(0, len(self.target_spaces))
        self.target_space = self.target_spaces[0]
        self.manipulated_obj = self.object_list[0].manipulated_part
        try_times = 0
        while len(self.temporary_waypoints)==0:
            self.sample_method()
            for _ in range(5):
                self.pyrep.step()
            self.register_success_conditions([DetectedCondition(self.manipulated_obj, self.target_space.successor)])
            init_states = self.get_state()[0]
            self.target_space.set_target(self.manipulated_obj, try_ik_sampling=True, release=True)
            MoveTask0 = T1_MoveObjectGoal(self.robot, self.pyrep, self.target_space, self.taks_base, fail_times=10)
            GraspTask0 = T0_ObtainControl(self.robot, self.pyrep, self.manipulated_obj,self.taks_base,\
                        next_task_fuc=MoveTask0.get_path, try_times=20)
            if try_times<100:
                waypoints = GraspTask0.get_path(try_ik_sampling=False)
            else:
                waypoints = GraspTask0.get_path(try_ik_sampling=True)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            self.robot.reset()
            self.pyrep.set_configuration_tree(init_states)
            try_times+= 1
        for i,waypoint in enumerate(self.temporary_waypoints):
            waypoint.set_name('waypoint{}'.format(i))
        description = 'Pick up {} and place it into {}.'.format(self.manipulated_obj.descriptions, self.target_space.target_space_descriptions)
        return [description]
    
    def is_static_workspace(self) -> bool:
        return True

    def load(self, ttms_folder=None):
        if Shape.exists('pick_cube'):
            return Dummy('pick_cube')
        ttm_file = os.path.join(ttms_folder, 'pick_cube.ttm')
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
    
    def get_base(self) -> Dummy:
        self._base_object = Dummy('pick_cube')
        return self._base_object
    
    def import_objects(self, num=2):
        if not hasattr(self, "model_path"):
            model_path = self.model_dir+"cube/cube_normal/cube_normal.ttm"
        else:
            model_path = self.model_dir+self.model_path
        for i in range(num):
            cube = VLM_Object(self.pyrep, model_path, i)
            cube.set_parent(self.taks_base)
            self.object_list.append(cube)
        self.register_graspable_objects(self.object_list)
    
    def sample_method(self):
        satisfied = False
        while not satisfied:
            self.spawn_space.clear()
            for space in self.target_spaces:
                self.spawn_space.sample(Shape(space.focus_obj_id), min_distance=0.25)
            # self.spawn_space.sample(Shape('small_container0'))
            # self.spawn_space.sample(Shape('small_container1'), min_distance=0.25)
            if hasattr(self, "get_relative_pos"):
                comparted_target = self.target_spaces[1]
                target_relative_pose = get_relative_position_xy(Shape(comparted_target.focus_obj_id), Shape(self.target_space.focus_obj_id), self.robot.arm)
                if target_relative_pose != self.destination_target_relative:
                    continue
                self.target_space.target_space_descriptions = f"the {target_relative_pose} container"
                # target_space0_pos = get_relative_position_xy(Shape("small_container1"), Shape('small_container0'), self.robot.arm)
                # target_space1_pos = get_relative_position_xy(Shape("small_container0"), Shape('small_container1'), self.robot.arm)
                # self.target_space0.target_space_descriptions = f"the {target_space0_pos} container"
                # self.target_space1.target_space_descriptions = f"the {target_space1_pos} container"
                # if self.target_space.target_space_descriptions != f"the {self.destination_target_relative} container":
                #     continue
            for obj in self.object_list:
                self.spawn_space.sample(obj, min_distance=0.1)
            if hasattr(self, "get_relative_pos"):
                comparted_obj = self.object_list[1]
                object_relative_pose = get_relative_position_xy(comparted_obj.manipulated_part, self.manipulated_obj, self.robot.arm)
                if object_relative_pose != self.object_target_relative:
                    continue
                self.manipulated_obj.property["relative_pos"] = object_relative_pose
                self.manipulated_obj.descriptions = "the {} {}".format(self.manipulated_obj.property["relative_pos"], self.manipulated_obj.property["shape"])
            satisfied = True
            self.pyrep.step()
