from vlm.tasks.open_door_fridge import OpenDoorFridge, door_states
from amsolver.backend.conditions import JointCondition
import numpy as np

class OpenDoorGrill(OpenDoorFridge):
    def import_model(self, model_dir):
        model_path = model_dir+"grill/grill1/grill1.ttm"
        self._ignore_collisions = True
        return model_path
    
    def door_setting(self, index):
        if index in [0,1]:
            init_state = "open"
            self.door_joint.set_joint_position(np.deg2rad(60), True)
        else:
            init_state = "close"
            self.door_joint.set_joint_position(np.deg2rad(0), True)
        sub_index = index%2
        goal_state = door_states[init_state][sub_index]
        if "Slightly" in goal_state:
            detect_angle = 25
            goal_angel = 30
            detect_bound = 40
        else:
            detect_angle = 45
            detect_bound = 60
            if "open" in goal_state:
                goal_angel = 50
            else:
                goal_angel = 10
        self.register_success_conditions([JointCondition(self.door_joint, np.deg2rad(detect_angle), np.deg2rad(detect_bound))])
        return goal_angel, goal_state