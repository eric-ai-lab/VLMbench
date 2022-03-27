import itertools
from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import scale_object, select_color
from vlm.tasks.pick_cube import PickCube

relative_pos_list = list(itertools.product(["left", "right", "front", "rear"], repeat=2))
class PickCubeRelative(PickCube):
    def init_task(self) -> None:
        super().init_task()
        self.get_relative_pos=True

    def init_episode(self, index: int) -> List[str]:
        color_index = np.random.choice(len(colors), len(self.object_list), replace=True)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])

        target_space_colors = np.random.choice(len(colors), len(self.target_spaces), replace=True)
        for i, target_space in enumerate(self.target_spaces):
            Shape(target_space.focus_obj_id).set_color(colors[target_space_colors[i]][1])

        self.object_target_relative, self.destination_target_relative = relative_pos_list[index]
        return super().init_episode(index)
    
    def variation_count(self) -> int:
        return len(relative_pos_list)
