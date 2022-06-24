# VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation

![task image missing](readme_files/teaser.svg)

**VLMbench** is a robotics manipulation benchmark, which contains various language instructions on categorized robotic manipulation tasks. In this work, we aim to fill the blank of the last mile of embodied agents---object manipulation by following human guidance, e.g., “move the red mug next to the box while keeping it upright.” VLMbench is the first benchmark that ***compositional designs*** for vision-and-language reasoning on manipulations and ***categorizes the manipulation tasks*** from the perspectives of task constraints. Meanwhile, we introduce an Automatic Manipulation Solver (**AMSolver**), where modular rule-based task templates are created to automatically generate robot demonstrations with language instructions, consisting of diverse object shapes and appearances, action types, and motion constraints.  [Click here for website and paper.](https://sites.google.com/ucsc.edu/vlmbench/home)

This repo include the implementaions of AMSolver, VLMbench, and 6D-CLIPort.

## AMSolver Install
Users can use AMSolver to run the current tasks in the VLMbench or build new tasks. In order to run the AMSolver, you should install [Coppliasim 4.1.0](https://www.coppeliarobotics.com/previousVersions) and [PyRep](https://github.com/stepjam/PyRep) first. Then, lets install AMSolver:

```bash
pip install -r requirements.txt
pip install -r cliport/requirements.txt
pip install .
```

Then, copy the simAddOnScript_PyRep.lua in current folder into the Coppliasim folder:
```bash
mv ./simAddOnScript_PyRep.lua /Path/To/Coppliasim
```
<!-- (meshlab for import new model; copy new simAddOnScript_PyRep.lua to Coppeliasim) -->

<!-- In the vlm folder, we have predefined some task categories and instance tasks for VLMbench. If you want to customize your own task, the scripts in the tools folder can be helpful. -->

## VLMbench Baselines

The precollected dataset can be found at here: [Dataset](https://drive.google.com/drive/folders/1Qx_2_ePIqf_Z6SnpPkocUiPgFeCfePQh?usp=sharing). The dataset is under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

The pretrained models can be found at here: [Model](https://drive.google.com/drive/folders/1yFbWhP2iHQvY04q8LNmrpT6_5ctTcZDk?usp=sharing)

To test pretrained 6D-CLIPort models:
```bash
python vlm/scripts/cliport_test.py --data_folder /Path/to/VLMbench/Dataset/test --checkpoints_folder /Path/to/Pretained/Models
```

To train new 6D-CLIPort models:
```bash
python vlm/scripts/train_baselines.py --data_dir /Path/to/VLMbench/Dataset --train_tasks TASK_NEED_TO_TRAIN
```

## Generate Customized Demonstrations

To generate new demonstrations for training and validation, users can set the output data directory in ***save_path*** parameter and run :

```bash
python tools/dataset_generator_NLP.py
```

Meanwhile, the test configurations can be generated by running:
```bash
python tools/test_config_generator.py
```
## Add new objects and tasks
All object models are saved in *vlm/object_models*. To import new objects into vlmbench, users can use "vlm/object_models/save_model.py". We recommand users first save the object models as a coppeliasim model file (.ttm), then use the extra_from_ttm function inside the save_model.py. More examples can be found in save_model.py.

All tasks templates in the current vlmbench can be found in *vlm/tasks*. To generate new task templates, users can use "tools/task_builder_NLP".py for basic task template generation. Then, the varations of the task can be written as the child classes of the basic task template. More details can refer the codes of *vlm/tasks*.