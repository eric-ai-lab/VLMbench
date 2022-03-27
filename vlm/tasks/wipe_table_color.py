from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import select_color
from vlm.tasks.wipe_table import WipeTable

class WipeTableColor(WipeTable):
    def init_task(self) -> None:
        # self.model_num = 3
        return super().init_task()
        
    def init_episode(self, index: int) -> List[str]:
        color_names, rgbs = select_color(index, len(self.target_list)-1)
        for i, target in enumerate(self.target_list):
            target.set_color(rgbs[i])
            target.target_space_descriptions= "the {} area".format(color_names[i])

        return super().init_episode(index)
    
    def variation_count(self) -> int:
        return len(colors)