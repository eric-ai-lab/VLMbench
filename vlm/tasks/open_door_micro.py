from vlm.tasks.open_door_fridge import OpenDoorFridge

class OpenDoorMicro(OpenDoorFridge):
    def import_model(self, model_dir):
        model_path = model_dir+"microwave/microwave1/microwave1.ttm"
        return model_path