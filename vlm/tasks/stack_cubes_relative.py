from typing import List
import numpy as np
import itertools
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.unit_tasks import VLM_Object
from amsolver.const import colors
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task
from amsolver.backend.utils import get_relative_position_xy, get_sorted_grasp_pose, scale_object, select_color
from vlm.tasks.stack_cubes import StackCubes
from pyrep.const import ObjectType, PrimitiveShape

object_dict = {
            "star":{
                "path":"star/star_normal/star_normal.ttm"
            },
            "moon":{
                "path":"moon/moon_normal/moon_normal.ttm"
            },
            "triangular":{
                "path":"triangular/triangular_normal/triangular_normal.ttm"
            },
            "cylinder":{
                "path":"cylinder/cylinder_normal/cylinder_normal.ttm"
            },
            "cube":{
                "path":"cube/cube_basic/cube_basic.ttm"
            }
        }
relative_pos_list = list(itertools.product(["left", "right", "front", "rear"], repeat=2))
class StackCubesRelative(StackCubes):

    def init_task(self) -> None:
        self.model_num = 2
        return super().init_task()

    def init_episode(self, index: int) -> List[str]:
        self.object_target_relative0, self.object_target_relative1 = relative_pos_list[index]
        random_objs = np.random.choice(list(self.shape_lib.keys()), 2, replace=False)
        self.select_obj0 = self.shape_lib[random_objs[0]]
        self.select_obj1 = self.shape_lib[random_objs[1]]
        self.cube_list = [self.select_obj0[0], self.select_obj1[0]]

        self.cube_num = len(self.cube_list)
        color_index = np.random.choice(len(colors), len(self.cube_list), replace=True)
        for i, obj in enumerate(self.cube_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])

        return super().init_episode(index)

    def is_static_workspace(self) -> bool:
        return True

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(relative_pos_list)
    
    def import_objects(self, num):
        object_numbers = [2]*len(object_dict)
        self.shape_lib = {obj:[] for obj in object_dict}
        all_objects = []
        for obj, num in zip(object_dict, object_numbers):
            for i in range(num):
                model = VLM_Object(self.pyrep, self.model_dir+object_dict[obj]["path"], i)
                relative_factor = scale_object(model, 1.25)
                if abs(relative_factor-1)>1e-2:
                    local_grasp_pose = model.manipulated_part.local_grasp
                    local_grasp_pose[:, :3, 3] *= relative_factor
                    model.manipulated_part.local_grasp = local_grasp_pose
                # model.set_model(False)
                model.set_parent(self.taks_base)
                model_bbox = model.get_bounding_box()
                x, y = model_bbox[1]-model_bbox[0]+0.02,  model_bbox[3]-model_bbox[2]+0.02
                model_target = Shape.create(PrimitiveShape.CUBOID, [x,y,0.02], respondable=False, static=True, renderable=False)
                # model_target.set_parent(self.taks_base)
                model_target._is_plane = True
                model_target.set_transparency(0)
                model.target = model_target
                model.set_position([0,0,0])
                self.shape_lib[obj].append(model)
                all_objects.append(model)
        self.register_graspable_objects(all_objects)
    
    def sample_method(self):
        satisfied = False
        while not satisfied:
            self.spawn_space.clear()
            for obj in self.select_obj0:
                self.spawn_space.sample(obj, min_distance=0.1)
            
            object_relative_pose = get_relative_position_xy(self.select_obj0[1], self.select_obj0[0], self.robot.arm)
            if object_relative_pose != self.object_target_relative0:
                continue
            obj = self.select_obj0[0].manipulated_part
            obj.descriptions = "the {} {}".format(object_relative_pose, obj.property["shape"]) 

            for obj in self.select_obj1:
                self.spawn_space.sample(obj, min_distance=0.1)
            object_relative_pose = get_relative_position_xy(self.select_obj1[1], self.select_obj1[0], self.robot.arm)
            if object_relative_pose != self.object_target_relative1:
                continue
            obj = self.select_obj1[0].manipulated_part
            obj.descriptions = "the {} {}".format(object_relative_pose, obj.property["shape"]) 
            satisfied = True
            self.pyrep.step()
