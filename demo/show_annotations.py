import argparse
import os.path as osp

from mmdet.models.utils import mask2ndarray

from mmdet.structures.bbox import BaseBoxes
from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar
from mmrotate.registry import DATASETS, VISUALIZERS
from mmrotate.utils import register_all_modules

def main():
    config_file = 'configs/two_stream/two_stream-3x-dronevehicle.py'
    cfg = Config.fromfile(config_file)
    # register all modules in mmdet into the registries
    register_all_modules()

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    item = dataset[0]
    rgb_img = item['inf'].permute(1, 2, 0).numpy()
    data_sample = item['data_samples'].numpy()
    gt_instances = data_sample.gt_instances
    img_path = osp.basename(item['data_samples'].inf_path)

    out_file = osp.join(
            'demo/out_dir',
            osp.basename(img_path))
    rgb_img = rgb_img[..., [2, 1, 0]]  # bgr to rgb
    gt_bboxes = gt_instances.get('bboxes', None)
    if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
        gt_instances.bboxes = gt_bboxes.tensor
    gt_masks = gt_instances.get('masks', None)
    if gt_masks is not None:
        masks = mask2ndarray(gt_masks)
        gt_instances.masks = masks.astype(bool)
    data_sample.gt_instances = gt_instances
    
    visualizer.add_datasample(
            osp.basename(img_path),
            rgb_img,
            data_sample,
            show=False,
            out_file=out_file)

if __name__ == '__main__':
    main()