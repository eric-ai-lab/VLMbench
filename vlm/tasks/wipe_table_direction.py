from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import get_relative_position_xy, select_color
from vlm.tasks.wipe_table import WipeTable

direction_list = ["horizontal", "vertical"]
class WipeTableDirection(WipeTable):
    def init_task(self) -> None:
        super().init_task()
        self._task_init_states = []
        for area in self.target_list:
            self._task_init_states.append(area.get_pose())
        
    def init_episode(self, index: int) -> List[str]:
        self.target_direction = direction_list[index]
        self.target_list[0].target_space_descriptions = f"the {self.target_direction} area"
        target_space_colors = np.random.choice(len(colors), len(self.target_list), replace=True)
        for i, target in enumerate(self.target_list):
            target.set_color(colors[target_space_colors[i]][1])

        return super().init_episode(index)
    
    def variation_count(self) -> int:
        return len(direction_list)
    
    def sample_method(self):
        for area, pose in zip(self.target_list, self._task_init_states):
            area.set_pose(pose)
        self.spawn_space.clear()
        vertical_min_rotation, vertical_max_rotation = (0, 0, -np.pi/8), (0, 0, np.pi/8)
        if np.random.uniform()<0.5:
            horizontal_min_rotation, horizontal_max_rotation = (0, 0, -5*np.pi/8), (0, 0, -3*np.pi/8)
        else:
            horizontal_min_rotation, horizontal_max_rotation = (0, 0, 3*np.pi/8), (0, 0, 5*np.pi/8)
        if self.target_direction == "horizontal":
            t0_min_rotation, t0_max_rotation = horizontal_min_rotation, horizontal_max_rotation
            t1_min_rotation, t1_max_rotation = vertical_min_rotation, vertical_max_rotation
        else:
            t0_min_rotation, t0_max_rotation = vertical_min_rotation, vertical_max_rotation
            t1_min_rotation, t1_max_rotation = horizontal_min_rotation, horizontal_max_rotation
        
        self.spawn_space.sample(self.target_list[0], min_distance=0.1, 
                            min_rotation=t0_min_rotation, max_rotation=t0_max_rotation)
        self.spawn_space.sample(self.target_list[1], min_distance=0.1, 
                            min_rotation=t1_min_rotation, max_rotation=t1_max_rotation)
            
        for obj in self.object_list:
            self.spawn_space.sample(obj, min_distance=0.1)