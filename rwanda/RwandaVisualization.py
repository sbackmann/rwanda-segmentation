import numpy as np
import argparse
import os
from collections import Sequence
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from torch.cuda import is_available

import mmcv
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root

from mmdet.apis import init_detector, inference_detector, show_result_pyplot


class RwandaVisualization:
    
    def __init__(self,
                 cfg,
                 checkpoint,
                 subset="val"):
        
        self.skip_type = ['DefaultFormatBundle', 'Normalize', 'Collect']
        cfg = replace_cfg_vals(cfg)

        update_data_root(cfg)

        if not (cfg.model.type == "MaskRCNN" and cfg.model.backbone.depth == 50):
            cfg.data.train = cfg.data.train.dataset
        if subset == "val":
            cfg.data.train.ann_file = cfg.data.val.ann_file
            cfg.data.train.img_prefix = cfg.data.val.img_prefix
        elif subset == "test":
            cfg.data.train.ann_file = cfg.data.test.ann_file
            cfg.data.train.img_prefix = cfg.data.test.img_prefix

        data_cfg = cfg.data.train
        while 'dataset' in data_cfg and data_cfg[
                    'type'] != 'MultiImageMixDataset':
                data_cfg = data_cfg['dataset']

        if isinstance(data_cfg, Sequence):
            [self._skip_pipeline_steps(c) for c in data_cfg]
        else:
            self._skip_pipeline_steps(data_cfg)

        #if 'gt_semantic_seg' in cfg.train_pipeline[-1]['keys'] or 'RandomFlip' in cfg.train_pipeline[-1]['keys']:
        cfg.data.train.pipeline = [
            p for p in cfg.data.train.pipeline if p['type'] not in ['SegRescale', 'RandomFlip']
        ]       
        
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.dataset = build_dataset(cfg.data.train)
        
        if is_available():
            device='cuda:0'
        else:
            warnings.warn("Warning: No cuda device found. Using CPU.")
            device='cpu'
        self.model = init_detector(self.cfg, self.checkpoint, device=device)


    def _skip_pipeline_steps(self, config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in self.skip_type
        ]
    
    
    def save_ground_truths(self, out_dir):
        """Save all image patches of ``self.subset`` including ground truth images.
        
            Args:
                out_dir (str): Path to the directory where images are saved. If non-existent, it will be created.        
        """

        os.makedirs(out_dir, exist_ok=True)
        if len(os.listdir(out_dir)) <= 1:
            print(f"Saving ground truth images into {out_dir}.")
            progress_bar = mmcv.ProgressBar(len(self.dataset))
            for item in self.dataset:
                self._plot_ground_truth(item, show=False, out_dir=out_dir, wait_time=2)
                progress_bar.update()
        else:
            print(f"Ground truth images already existing in {out_dir}.")

            
    def save_inference_results(self, out_dir):
        """Save inference results for all image patches of ``self.subset``.
        
            Args:
                out_dir (str): Path to the directory where results are saved. If non-existent, it will be created.        
        """
        
        os.makedirs(out_dir, exist_ok=True)
        if len(os.listdir(out_dir)) <= 1:
            print(f"\nSaving inference results into {out_dir}.")
            progress_bar = mmcv.ProgressBar(len(self.dataset))
            for item in self.dataset:
                self.plot_inference_results(item, show=False, out_dir=out_dir, wait_time=2)
                progress_bar.update()
        else:
            print(f"Inference results already existing in {out_dir}.")
                                            
            
    def eval_image(self, img=None, save_fig=False, out_file='./visualization/gt_inf_comparison.png'):
        """Plot ground truth and prediction for one image.
        
            Args:
                img (str, optional): Either none (random image) or file name (not path) of the image to be shown.
                save_fig (bool, optional): Whether or not the figure is saved.
                out_file (str, optional): save file path if save_fig is set to True.       
        """
        
        if img is None:
            rnd = np.random.randint(len(self.dataset))
            img = self.dataset[rnd]
        else:
            imgs = [d for d in self.dataset if d['img_info']['filename'] == img]
            if len(imgs) == 1:
                img = imgs[0]
            else:
                raise ValueError(f"{img} cannot be found.")
        gt = self._plot_ground_truth(img, show=False, out_dir=None, wait_time=0)
        inf = self.plot_inference_results(img, show=False)
        print(f"Image: {img['filename']}")
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,8))
        #print(gt)
        #print(inf)
        #print(gt.shape, inf.shape)
        ax1.axis("off")
        ax2.axis("off")
        ax1.set_title("Ground Truth")
        ax2.set_title("Prediction")
        ax1.imshow(gt)
        ax2.imshow(inf)
        if save_fig:
            plt.savefig(out_file)
            print(f"Figure saved in {out_file}.") 
        
    
    def _plot_ground_truth(self, image, show, out_dir=None, wait_time=0):
        """Helper function to create ground truth image.
        
            Args:
                image (dict): one image from the built dataset.
                show (bool): Whether or not the created plot is shown.
                out_dir (str, optional): The path to the output directory. If not None, the result is saved.
                wait_time (float):  Value of waitKey param.
        """
        
        if out_dir:
            out_file = os.path.join(out_dir,
                            Path(image['filename']).name)
        else:
            out_file = None

        gt_bboxes = image['gt_bboxes']
        gt_labels = image['gt_labels']
        gt_masks = image.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)

        gt_seg = image.get('gt_semantic_seg', None)
        if gt_seg is not None:
            pad_value = 255  # the padding value of gt_seg
            sem_labels = np.unique(gt_seg)
            all_labels = np.concatenate((gt_labels, sem_labels), axis=0)
            all_labels, counts = np.unique(all_labels, return_counts=True)
            stuff_labels = all_labels[np.logical_and(counts < 2,
                                                     all_labels != pad_value)]
            stuff_masks = gt_seg[None] == stuff_labels[:, None, None]
            gt_labels = np.concatenate((gt_labels, stuff_labels), axis=0)
            gt_masks = np.concatenate((gt_masks, stuff_masks.astype(np.uint8)),
                                      axis=0)
            # If you need to show the bounding boxes,
            # please comment the following line
            gt_bboxes = None

        img = imshow_det_bboxes(
                image['img'],
                gt_bboxes,
                gt_labels,
                gt_masks,
                class_names=self.dataset.CLASSES,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                bbox_color=self.dataset.PALETTE,
                text_color=(200, 200, 200),
                mask_color=self.dataset.PALETTE)
        return img
    
        
    def plot_inference_results(self, image, show=True, out_dir=None, wait_time=0):
        """Run inference on one image and create result.
        
            Args:
                image (dict or str): one image from the built dataset or the path to the image.
                show (bool): Whether or not the created plot is shown.
                out_dir (str, optional): The path to the output directory. If not None, the result is saved.
                wait_time (float):  Value of waitKey param.
        """
        
        out_file = None
        
        if isinstance(image, str):
            img = mmcv.imread(image)
        else:
            img = image['img']
            if out_dir:
                out_file = os.path.join(out_dir,
                            Path(image['filename']).name)

        self.model.cfg = self.cfg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = inference_detector(self.model, img)
        if hasattr(self.model, 'module'):
            self.model = self.model.module
        return_img = self.model.show_result(
                    img,
                    result,
                    score_thr=0.3,
                    show=show,
                    wait_time=0,
                    win_name=None,
                    bbox_color=None,
                    text_color=(200, 200, 200),
                    mask_color=None,
                    out_file=out_file)
        if not show and not out_file:
            return return_img