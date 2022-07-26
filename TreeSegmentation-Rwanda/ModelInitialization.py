import os
import subprocess
import pickle
import re
from mmcv.utils import Config

class ModelInitialization:
    
    def __init__(self, experiment):
        self.experiment_dir = None
        experiments = os.listdir("./experiments")
        for directory in experiments:
            if directory.endswith(experiment):
                self.experiment_dir = directory
                break
                
        assert self.experiment_dir is not None, \
        "Experiment can not be found."
        
        self.experiment = experiment
        
        
        
        
    def load_pretrained_model(self, model_dir):
        """Downloads the pretrained models if they are not already downloaded.

        Args:
            model_dir (str): The path to the directory where the models are downloaded into and that is also used
                    check if the model is already loaded.
            model (str): The name of the model to load. Options are ``MaskR50`` , ``MaskR101``, ``CascadeR50``
                    and ``CascadeR101``.
        """
        model = re.search('^(.*)_', self.experiment).group(1)

        assert model in ["MaskR50", "MaskR101", "CascadeR50", "CascadeR101"]

        pretrained_files = {
            "MaskR50": "mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth",
            "MaskR101": "mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco_20210526_132339-3c33ce02.pth",
            "CascadeR50": "cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210707_002651-6e29b3a6.pth",
            "CascadeR101": "cascade_mask_rcnn_r101_caffe_fpn_mstrain_3x_coco_20210707_002620-a5bd2389.pth"}

        if os.path.exists(os.path.join(model_dir, pretrained_files[model])):
            print(f"{model} already downloaded.")
        else:
            print(f" Downloading pre-trained {model} weights. Please wait...")
            if model == "MaskR50":
                bash_command = f"wget https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/" \
                               f"mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/" \
                               f"mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth -P {model_dir}"
            elif model == "MaskR101":
                bash_command = f"wget https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/" \
                               f"mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco/" \
                               f"mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco_20210526_132339-3c33ce02.pth -P {model_dir}"
            elif model == "CascadeR50":
                bash_command = f"wget https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/" \
                               f"cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco/" \
                               f"cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210707_002651-6e29b3a6.pth -P {model_dir}"
            elif model == "CascadeR101":
                bash_command = f"wget https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/" \
                               f"cascade_mask_rcnn_r101_caffe_fpn_mstrain_3x_coco/" \
                               f"cascade_mask_rcnn_r101_caffe_fpn_mstrain_3x_coco_20210707_002620-a5bd2389.pth -P {model_dir}"

            subprocess.run(bash_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"{model} downloaded into {model_dir}.")
            
        return os.path.join(model_dir, pretrained_files[model])
    
    
    def load_experiment_weights(self):
        
        weight_files = {
            "MaskR50_-": "MaskR50_-_epoch15.pth",
            "MaskR101_-": "MaskR101_-_epoch5.pth",
            "MaskR50_1": "MaskR50_1_epoch33.pth",
            "MaskR50_2_": "MaskR50_2_epoch18.pth",
            "MaskR50_3": "MaskR50_3_epoch15.pth",
            "MaskR50_4": "MaskR50_4_epoch15.pth",
            "MaskR50_5": "MaskR50_5_epoch21.pth",
            "MaskR50_6": "MaskR50_6_epoch15.pth",
            "MaskR50_7": "MaskR50_7_epoch18.pth",
            "CascadeR50_-": "CascadeR50_-_epoch7.pth",
            "CascadeR101_-": "CascadeR101_-_epoch6.pth",
            "MaskR50_a": "MaskR50_a_epoch15.pth",
            "MaskR50_ab": "MaskR50_ab_epoch15.pth",
            "MaskR50_ac": "MaskR50_ac_epoch24.pth",
            "MaskR50_acd": "MaskR50_acd_epoch24.pth",
            "MaskR50_acde": "MaskR50_21_epoch21.pth",
            "CascadeR50_acd": "CascadeR50_acd_epoch5.pth"
        }
        if os.path.exists(os.path.join("./experiments", self.experiment_dir, weight_files[self.experiment])):
            print("Weights already downloaded.")
        else:
            print(f" Downloading trained {self.experiment} weights. Please wait...")
            repo_release = "https://github.com/sbackmann/rwanda-segmentation/releases/download/weights/"
            download_link = repo_release + weight_files[self.experiment]
            bash_command = f"wget {download_link} -P {os.path.join('./experiments', self.experiment_dir)}"
            subprocess.run(bash_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"{self.experiment} weights downloaded into {os.path.join('./experiments', self.experiment_dir)}")
            
        return os.path.join("./experiments", self.experiment_dir, weight_files[self.experiment])
        


    def load_config(self, anno_dir, img_dir, checkpoint):

        config_file = f"./experiments/{self.experiment_dir}/cfg.pkl"
        uses_oversampling = re.search('_.*c.*$', self.experiment)
        if uses_oversampling:
            prefix=""
        else:
            prefix="300_"
        #cfg = pickle.load(open(config_file,'rb'))
        with open(config_file, "rb") as f:
            cfg = pickle.load(f)
        if self.experiment.startswith("MaskR50"):
            cfg.data.train.ann_file = os.path.join(anno_dir, f"{prefix}train_annotation_coco.json")
            cfg.data.train.img_prefix = os.path.join(img_dir, "train")
        else:
            cfg.data.train.dataset.ann_file = os.path.join(anno_dir, f"{prefix}train_annotation_coco.json")
            cfg.data.train.dataset.img_prefix = os.path.join(img_dir, "train")
        cfg.data.val.ann_file = os.path.join(anno_dir, "val_annotation_coco.json")
        cfg.data.val.img_prefix = os.path.join(img_dir, "val")
        cfg.data.test.ann_file = os.path.join(anno_dir, "test_annotation_coco.json")
        cfg.data.test.img_prefix = os.path.join(img_dir, "test")
        cfg.load_from = checkpoint
        cfg = Config(cfg._cfg_dict, filename=f"../MMDetection/{cfg.filename[3:]}")
        print(f"Loaded configuration for {self.experiment}.")
        return cfg
    
    @staticmethod
    def print_experiments():
        experiments = os.listdir('./experiments')
        experiments.sort()
        for directory in experiments:
            experiment = re.search('^\S{2,3}_(.*)', directory)
            if experiment:
                print(experiment.group(1))