import warnings
import copy
from argparse import ArgumentParser
from typing import Optional, Sequence, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import mmcv
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet.structures import DetDataSample, SampleList
from mmdet.evaluation import get_classes
from mmdet.utils import get_test_pipeline_cfg

from mmrotate.registry import VISUALIZERS, MODELS, DATASETS
from mmrotate.utils import register_all_modules

def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = 'none',
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to none.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.rgb_backbone:
        config.model.rgb_backbone.init_cfg = None
        config.model.inf_backbone.init_cfg = None

    scope = config.get('default_scope', 'mmrotate')
    if scope is not None:
        init_default_scope(config.get('default_scope', 'mmrotate'))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter('once')
        warnings.warn('checkpoint is None, use COCO classes by default.')
        model.dataset_meta = {'classes': get_classes('coco')}
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})

        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            # mmdet 3.x, all keys should be lowercase
            model.dataset_meta = {
                k.lower(): v
                for k, v in checkpoint_meta['dataset_meta'].items()
            }
        elif 'CLASSES' in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta['CLASSES']
            model.dataset_meta = {'classes': classes}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

    # Priority:  args.palette -> config -> checkpoint
    if palette != 'none':
        model.dataset_meta['palette'] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

def inference_two_stream(
    model: nn.Module,
    rgb_imgs: ImagesType,
    inf_imgs: ImagesType,
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector
    Args:
        model (nn.Module): The loaded detector.
        rgb_imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        inf_imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.
    """
    if isinstance(rgb_imgs, (list, tuple)):
        is_batch = True
    else:
        rgb_imgs = [rgb_imgs]
        inf_imgs = [inf_imgs]
        is_batch = False
    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(rgb_imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'LoadPairedImageFromNDArray'
        test_pipeline = Compose(test_pipeline)
    
    result_list = []
    for i, (rgb_img, inf_img) in enumerate(zip(rgb_imgs, inf_imgs)):
        # prepare data
        if isinstance(rgb_img, np.ndarray):
            # TODO remove img_id.
            data_ = dict(rgb_img=rgb_img, inf_img=inf_img, img_id=0)
        else:
            # TODO remove img_id
            data_ = dict(rgb_path=rgb_img, inf_path=inf_img, img_id=0)
        
        if text_prompt:
            data_['text'] = text_prompt
            data_['custom_entities'] = custom_entities
        
        # build the data pipeline
        data_ = test_pipeline(data_)
        # simulate the pseduo_collate
        data_['rgb'] = [data_['rgb']]
        data_['inf'] = [data_['inf']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            # DetDataSample, contains `pred_instances`
            results = model.test_step(data_)[0]

        result_list.append(results)
    
    if not is_batch:
        return result_list[0]
    else:
        return result_list

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('rgb_img', help='Image file')
    parser.add_argument('inf_img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def main(args):
    # register all modules in mmrotate into the registries
    register_all_modules()
    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)
    
    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # test a RGB-IR pair image
    # only test one image
    result = inference_two_stream(model, args.rgb_img, args.inf_img)

    # show the results
    rgb_img = mmcv.imread(args.rgb_img)
    rgb_img = mmcv.imconvert(rgb_img, 'bgr', 'rgb')
    inf_img = mmcv.imread(args.inf_img)
    inf_img = mmcv.imconvert(inf_img, 'bgr', 'rgb')

    visualizer.add_datasample(
        'result',
        inf_img,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=args.out_file,
        pred_score_thr=args.score_thr)

if __name__ == '__main__':
    args = parse_args()
    main(args)


