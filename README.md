# Detection and Segmentation of Tree Instances on a Rwandan Satellite Dataset


## Introduction
Rwanda-Instance provides the code for the Bachelor Thesis _Detection and Segmentation of Tree Instances on a Rwandan Satellite Dataset_. It uses [MMDetection](https://github.com/open-mmlab/mmdetection) 2.25.0, an open source object detection toolbox based on PyTorch.

The code that was added in the course of this thesis can be found in the folder ``TreeSegmentation-Rwanda``. The directory MMDetection contains the MMDetection model library, parts of its code were modified to better work with the project. Additionally, the folder ``cocoapi`` is a modified version (e.g., an increased number of detections for AP calculation) of https://github.com/cocodataset/cocoapi.


## Installation

**Step 1:** Install PyTorch 1.10 with the PyTorch CUDA version matching the compiling CUDA version. E.g., for CUDA 11.1:
```shell
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
**Step 2:** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).
```shell
pip install -U openmim
mim install mmcv-full==1.5.0
```
**Step 3:** Clone repository and install MMDetection.
```shell
git clone https://github.com/sbackmann/rwanda-instance
cd rwanda-instance
pip install -v -e MMDetection
pip install cocoapi/PythonAPI
```
## Getting Started
After finishing the installation steps, the satellite images and the annotations should be pasted into the respective folders (``TreeSegmentation-Rwanda/data/Training_Images_RGB`` for the images and ``TreeSegmentation-Rwanda/data/Training_tree_polygons`` for the annotation files including the ``.shp`` file).

Within the ``TreeSegmentation-Rwanda`` folder, two notebooks are available. ``Training.ipynb`` enables to reproduce the training that led to the models evaluated in the thesis. ``Testing.ipynb`` lets the user inference with the trained model weights from the thesis (or using their own) and also provides the means to evaluate the model metrics for veryfing the AP results.

All the experiments are located in the directory ``TreeSegmentation-Rwanda/experiments`` and are named following the pattern {Model}{Backbone}_{ConfigurationChanges}. For the configuration changes, see the tables specifying the change keys in the thesis. The model weights are released [here](https://github.com/sbackmann/rwanda-instance/releases/tag/v2.25.0) and will be downloaded automatically if the respective experiment is selected.

## License

MMDetection is released under the [Apache 2.0 license](LICENSE). To comply with MMDetection's Apache 2.0 License, all those files of MMDetection that were modified contain a comment, stating that they were changed. The thesis itself is released under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
