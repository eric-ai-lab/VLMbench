# VLMbench: A Benchmark for Vision-and-Language Manipulation

![task image missing](readme_files/tasks.svg)

**VLMbench** is a robotics manipulation benchmark, which contains various language instructions on categorized robotic manipulation tasks. In this work, we aim to fill the blank of the last mile of embodied agents---object manipulation by following human guidance, e.g., “move the red mug next to the box while keeping it upright.” Meanwhile, we introduce an Automatic Manipulation Solver (**AMSolver**), where modular rule-based task templates are created to automatically generate robot demonstrations with language instructions, consisting of diverse object shapes and appearances, action types, and motion constraints.  [Click here for website and paper.](https://sites.google.com/corp/view/rlbench)

The implementaions of AMSolver, VLM, and 6D-CLIPort.

## AMSolver Install
Users can use AMSolver to run the current tasks in the VLMbench or build new tasks. In order to run the AMSolver, you should install [PyRep](https://github.com/stepjam/PyRep) first. Then, lets install AMSolver:

```bash
pip install -r requirements.txt
pip install .
```
(meshlab for import new model; copy new simAddOnScript_PyRep.lua to Coppeliasim)

In the vlm folder, we have predefined some task categories and instance tasks for VLMbench. If you want to customize your own task, the scripts in the tools folder can be helpful.

## VLMbench Baselines

The precollected dataset can be found at here: [Dataset](https://drive.google.com/drive/folders/17dEJrIIdlDsDF6T2rn04y7Yy8mUpKfCK?usp=sharing)

The pretrained models can be found at here: [Model](https://drive.google.com/drive/folders/1yFbWhP2iHQvY04q8LNmrpT6_5ctTcZDk?usp=sharing)

To train new 6D-CLIPort models:
```bash
python vlm/scripts/train_baselines.py
```

To test pretrained 6D-CLIPort models:
```bash
python vlm/scripts/cliport_test.py
```
