from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import scale_object, select_color
from vlm.tasks.wipe_table import WipeTable
from pyrep.const import ObjectType, PrimitiveShape

shape_list = ["rectangle", "round"]
DIRT_POINTS = 50
class WipeTableShape(WipeTable):
    def init_task(self) -> None:
        # self.model_num = 3
        return super().init_task()
        
    def init_episode(self, index: int) -> List[str]:
        target_size = shape_list[index]
        if target_size == "rectangle":
            self.target_list = [self.area_list[0], self.area_list[1]]
        else:
            self.target_list = [self.area_list[1], self.area_list[0]]

        target_space_colors = np.random.choice(len(colors), len(self.target_list), replace=True)
        for i, target in enumerate(self.target_list):
            target.set_color(colors[target_space_colors[i]][1])

        return super().init_episode(index)
    
    def variation_count(self) -> int:
        return len(shape_list)
    
    def create_area(self):
        self.area_list = []
        dirt_area0 = Shape.create(PrimitiveShape.CUBOID, [0.3, 0.1, 0.001],
                        respondable=False, static=True, renderable=True)
        dirt_area0.set_name(f"dirt_area0")
        dirt_area0.set_parent(self.taks_base)
        dirt_area0.target_space_descriptions = "the rectangle area"
        self.area_list.append(dirt_area0)

        dirt_area1 = Shape.create(PrimitiveShape.CYLINDER, [0.2, 0.1, 0.001],
                        respondable=False, static=True, renderable=True)
        dirt_area1.set_name(f"dirt_area1")
        dirt_area1.set_parent(self.taks_base)
        dirt_area1.target_space_descriptions = "the round area"
        self.area_list.append(dirt_area1)
    
    def _place_dirt(self, space):
        target_pose = space.get_matrix()
        target_size = space.get_bounding_box()
        # spawn = SpawnBoundary([space])
        if self.variation_index == 0:
            step = (target_size[1]-target_size[0])/DIRT_POINTS
            for i in range(DIRT_POINTS):
                spot = Shape.create(type=PrimitiveShape.CUBOID,
                                    size=[.005, .005, .001],
                                    mass=0, static=True, respondable=False,
                                    renderable=False,
                                    color=[0.58, 0.29, 0.0])
                spot.set_parent(space)
                delta_y = np.random.normal(scale=0.005)
                related_pos = target_pose.dot(np.array([target_size[0]+step*i, delta_y, 0.001, 1]))
                spot.set_position(related_pos[:3])
                # spawn.sample(spot, min_distance=0.00,
                #               min_rotation=(0.00, 0.00, 0.00),
                #               max_rotation=(0.00, 0.00, 0.00), place_above_plane=False)
                self.dirt_spots.append(spot.get_name())
        else:
            max_r = target_size[1]/2
            for i in range(DIRT_POINTS):
                spot = Shape.create(type=PrimitiveShape.CUBOID,
                                    size=[.005, .005, .001],
                                    mass=0, static=True, respondable=False,
                                    renderable=False,
                                    color=[0.58, 0.29, 0.0])
                spot.set_parent(space)
                r = np.random.uniform(0, max_r)
                theta = np.random.uniform(-np.pi, np.pi)
                x, y = r*np.cos(theta), r*np.sin(theta)
                related_pos = target_pose.dot(np.array([x, y, 0.001, 1]))
                spot.set_position(related_pos[:3])
                # spawn.sample(spot, min_distance=0.00,
                #               min_rotation=(0.00, 0.00, 0.00),
                #               max_rotation=(0.00, 0.00, 0.00), place_above_plane=False)
                self.dirt_spots.append(spot.get_name())