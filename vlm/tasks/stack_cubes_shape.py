from typing import List
import numpy as np
import os
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.unit_tasks import VLM_Object
from amsolver.const import colors
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task
from amsolver.backend.utils import scale_object, select_color
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

class StackCubesShape(StackCubes):

    def init_episode(self, index: int) -> List[str]:
        self.cube_list = [self.shape_lib[index]]
        other_obj_index = list(range(len(self.shape_lib)))
        other_obj_index.remove(index)
        distractor_number = np.random.randint(1,len(self.shape_lib))
        distractor_index = np.random.choice(other_obj_index, distractor_number, replace=False)

        for i in distractor_index:
            self.cube_list.append(self.shape_lib[i])

        self.cube_num = len(self.cube_list)
        color_index = np.random.choice(len(colors), len(self.cube_list), replace=True)
        for i, obj in enumerate(self.cube_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])
            obj.manipulated_part.descriptions = "the {}".format(obj.manipulated_part.property["shape"])
        # color_names, rgbs = select_color(index, 3)
        # for i, cube in enumerate(self.cube_list):
        #     Shape(cube.manipulated_part.visual).set_color(rgbs[i])
        #     cube.manipulated_part.descriptions = f"the {color_names[i]} cube"
        return super().init_episode(index)

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(object_dict)
    
    def is_static_workspace(self) -> bool:
        return True
        
    def import_objects(self, num):
        object_numbers = [1]*len(object_dict)
        self.shape_lib = []
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
                # model_target.set_parent(model)
                model_target._is_plane = True
                model_target.set_transparency(0)
                model.target = model_target
                model.set_position([0,0,0])
                self.shape_lib.append(model)
        self.register_graspable_objects(self.shape_lib)
