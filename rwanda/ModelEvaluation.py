import os
import time
import pickle
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)


def evaluate_rwanda(cfg, checkpoint, subset="val", out_dir=None):
    """Evaluate model on a subset and calculate AP.
    
    Args:
        cfg (dict): The model configuration.
        checkpoint (str): The path to the trained model weights.
        subset (str): The subset on which to evaluate the model.
        out_dir (str, optional): The path to the output directory. If not specified, the results won't be saved.
    """
    
    ## This code is to a large degree from tools/test.py and adapted to the dataset as well as notebook-based execution. 
    
    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.gpu_ids = [0]
    cfg.device = get_device()

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if subset == "val":
        cfg.data.test = cfg.data.val
    elif subset == "train":
        if not (cfg.model.type == "MaskRCNN" and cfg.model.backbone.depth == 50):
            cfg.data.train = cfg.data.train.dataset
        cfg.data.test.ann_file = cfg.data.train.ann_file
        cfg.data.test.img_prefix = cfg.data.train.img_prefix

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if out_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(os.path.abspath(out_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = os.path.join(out_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES


    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(model, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
                    
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
        ]:
            eval_kwargs.pop(key, None)
        kwargs = {}
        metrics=["bbox", "segm"]
        eval_kwargs.update(dict(metric=metrics, **kwargs))
        metric = dataset.evaluate(outputs, **eval_kwargs)
        metric_dict = dict(config=cfg, metric=metric)
        if out_dir is not None and rank == 0:
            print(f'\nwriting results to {out_dir}')
            mmcv.dump(metric_dict, json_file)