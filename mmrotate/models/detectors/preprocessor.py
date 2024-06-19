import random
from numbers import Number
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple, Union

from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from mmdet.structures import DetDataSample
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmrotate.registry import MODELS

from mmrotate.testing import demo_mm_inputs, get_detector_cfg
from mmrotate.utils import register_all_modules

@MODELS.register_module()
class PairedImageDataPreprocessor(BaseDataPreprocessor):
    def __init__(self,
                 mean_rgb: Optional[Sequence[Union[float, int]]] = None,
                 std_rgb: Optional[Sequence[Union[float, int]]] = None,
                 mean_inf: Optional[Sequence[Union[float, int]]] = None,
                 std_inf: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False):
        super().__init__(non_blocking)
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean_rgb is None) == (std_rgb is None), (
            'mean_rgb and std_rgb should be both None or tuple')
        assert (mean_inf is None) == (std_inf is None), (
            'mean_inf and std_inf should be both None or tuple')
        
        if mean_rgb is not None:
            assert len(mean_rgb) == 3 or len(mean_rgb) == 1, (
                '`mean_rgb` should have 1 or 3 values, to be compatible with '
                f'RGB or gray image, but got {len(mean_rgb)} values')
            assert len(std_rgb) == 3 or len(std_rgb) == 1, (  # type: ignore
                '`std_rgb` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std_rgb)} values')  # type: ignore
            self._enable_normalize_rgb = True
            self.register_buffer('mean_rgb',
                                 torch.tensor(mean_rgb).view(-1, 1, 1), False)
            self.register_buffer('std_rgb',
                                 torch.tensor(std_rgb).view(-1, 1, 1), False)
        else:
            self._enable_normalize_rgb = False
        
        if mean_inf is not None:
            assert len(mean_inf) == 3 or len(mean_inf) == 1, (
                '`mean_inf` should have 1 or 3 values, to be compatible with '
                f'RGB or gray image, but got {len(mean_inf)} values')
            assert len(std_inf) == 3 or len(std_inf) == 1, (  # type: ignore
                '`std_inf` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std_inf)} values')  # type: ignore
            self._enable_normalize_inf = True
            self.register_buffer('mean_inf',
                                 torch.tensor(mean_inf).view(-1, 1, 1), False)
            self.register_buffer('std_inf',
                                 torch.tensor(std_inf).view(-1, 1, 1), False)
        else:
            self._enable_normalize_inf = False
        
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = self.cast_data(data)
        _batch_rgbs = data['rgb']
        _batch_infs = data['inf']
        ##################### rgb Tensor Preprocess #######################
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_rgbs, torch.Tensor):
            batch_rgbs = []
            for _batch_rgb in _batch_rgbs:
                # channel transform
                if self._channel_conversion:
                    _batch_rgb = _batch_rgb[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_rgb = _batch_rgb.float()
                # Normalization
                if self._enable_normalize_rgb:
                    if self.mean_rgb[0] == 3:
                        assert _batch_rgb.dim(
                        ) == 3 and _batch_rgb.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_rgb.shape}')
                    _batch_rgb = (_batch_rgb - self.mean_rgb) / self.std_rgb
                batch_rgbs.append(_batch_rgb)
            # Pack and stack RGB Tensor
            batch_rgbs = stack_batch(batch_rgbs, self.pad_size_divisor, 
                                     self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_rgbs, torch.Tensor):
            assert _batch_rgbs.dim() == 4, (
                'The input of `PairedImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_rgbs.shape}')
            if self._channel_conversion:
                _batch_rgbs = _batch_rgbs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_rgbs = _batch_rgbs.float()
            if self._enable_normalize_rgb:
                _batch_rgbs = (_batch_rgbs - self.mean_rgb) / self.std_rgb
            h, w = _batch_rgbs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_rgbs = F.pad(_batch_rgbs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')

        ##################### inf Tensor Preprocess #######################
        if is_seq_of(_batch_infs, torch.Tensor):
            batch_infs = []
            for _batch_inf in _batch_infs:
                # channel transform
                if self._channel_conversion:
                    _batch_inf = _batch_inf[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_inf = _batch_inf.float()
                # Normalization
                if self._enable_normalize_inf:
                    if self.mean_inf[0] == 3:
                        assert _batch_inf.dim(
                        ) == 3 and _batch_inf.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_inf.shape}')
                    _batch_inf = (_batch_inf - self.mean_inf) / self.std_inf
                batch_infs.append(_batch_inf)
            # Pack and stack RGB Tensor
            batch_infs = stack_batch(batch_infs, self.pad_size_divisor, 
                                     self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_infs, torch.Tensor):
            assert _batch_infs.dim() == 4, (
                'The input of `PairedImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_infs.shape}')
            if self._channel_conversion:
                _batch_infs = _batch_infs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_infs = _batch_infs.float()
            if self._enable_normalize_inf:
                _batch_infs = (_batch_infs - self.mean_inf) / self.std_inf
            h, w = _batch_infs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_infs = F.pad(_batch_infs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')

        data['rgb'] = batch_rgbs
        data['inf'] = batch_infs
        data.setdefault('data_samples', None)
        return data

@MODELS.register_module()
class PairedDetDataPreprocessor(PairedImageDataPreprocessor):
    """Image pre-processor for two stream detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It supports batch augmentations.
    2. It will additionally append batch_input_shape and pad_shape
    to data_samples considering the object detection task.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """
    def __init__(self,
                 mean_rgb: Sequence[Number] = None,
                 std_rgb: Sequence[Number] = None,
                 mean_inf: Sequence[Number] = None,
                 std_inf: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean_rgb=mean_rgb,
            std_rgb=std_rgb,
            mean_inf=mean_inf,
            std_inf=std_inf,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        rgbs, infs, data_samples = data['rgb'], data['inf'], data['data_samples']

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(rgbs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })
        
            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)
            
            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)
        
        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                rgbs, data_samples = batch_aug(rgbs, data_samples)
                infs, data_samples = batch_aug(infs, data_samples)

        return {'inputs': {"rgb":rgbs,'inf': infs}, 'data_samples': data_samples}

    
    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['rgb']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[3] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape
    
    def pad_gt_masks(self,
                     batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape,
                    pad_val=self.mask_pad_value)
    def pad_gt_sem_seg(self,
                       batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if 'gt_sem_seg' in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode='constant',
                    value=self.seg_pad_value)
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)

# if __name__ == '__main__':
    # register_all_modules()
    # data_preprocessor = dict(
    #     type='PairedDetDataPreprocessor',
    #     mean_rgb=[159.88, 162.22, 160.28],
    #     std_rgb=[56.96, 59.57, 63.11],
    #     mean_inf=[136.63, 136.63, 136.63],
    #     std_inf=[64.97, 64.97, 64.97],
    #     bgr_to_rgb=True,
    #     pad_size_divisor=32,
    #     boxtype2tensor=False,
    #     batch_augments=None
    # )

#     preprocessor = MODELS.build(data_preprocessor)

#     packed_inputs = demo_mm_inputs(2, [[3, 224, 224], [3, 224, 224]], 
#                                    use_box_type=True)
#     packed_inputs['rgb'] = packed_inputs['inputs']
#     packed_inputs['inf'] = packed_inputs['inputs']
#     packed_inputs.pop('inputs', None)

#     data = preprocessor(packed_inputs)

#     print(data)