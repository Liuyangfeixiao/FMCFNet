from typing import List, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F
from torch import Tensor, einsum
from einops import rearrange

from mmdet.models import BaseDetector
from mmdet.structures import OptSampleList, SampleList, DetDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from mmrotate.registry import MODELS

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

@MODELS.register_module()
class TwoStreamDetector(BaseDetector):
    """Using two modality features to densely predict
    bounding boxes on the output features of the backbone+neck
    """

    def __init__(self, 
                 rgb_backbone: ConfigType,
                 inf_backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.rgb_backbone = MODELS.build(rgb_backbone)
        self.inf_backbone = MODELS.build(inf_backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    def _init_fusion_modules(self):
        return None
    
    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
    def forward(self, inputs: Union[Dict, Tensor], 
                data_samples: OptSampleList = None, 
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
        
    def loss(self, batch_inputs: Dict,
             batch_data_samples: SampleList) -> dict:
        # (B, C, H, W)
        img_rgb = batch_inputs['rgb']
        img_inf = batch_inputs['inf']
        # Tuple(Tensor)
        x_rgb = self.extract_rgb_feat(img_rgb)
        x_inf = self.extract_inf_feat(img_inf)

        x = self.fuse_feats(x_rgb, x_inf)
        if self.with_neck:
            x = self.neck(x)
        losses = self.bbox_head.loss(x, batch_data_samples)

        return losses
    
    def predict(self, batch_inputs: Dict, 
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        # (B, C, H, W)
        img_rgb = batch_inputs['rgb']
        img_inf = batch_inputs['inf']
        # Tuple(Tensor)
        x_rgb = self.extract_rgb_feat(img_rgb)
        x_inf = self.extract_inf_feat(img_inf)

        x = self.fuse_feats(x_rgb, x_inf)
        if self.with_neck:
            x = self.neck(x)
        
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    def _forward(self, batch_inputs: Dict, 
                 batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        # (B, C, H, W)
        img_rgb = batch_inputs['rgb']
        img_inf = batch_inputs['inf']
        # Tuple(Tensor)
        x_rgb = self.extract_rgb_feat(img_rgb)
        x_inf = self.extract_inf_feat(img_inf)

        x = self.fuse_feats(x_rgb, x_inf)
        if self.with_neck:
            x = self.neck(x)
        
        results = self.bbox_head.forward(x)
        return results


    # TODO 将两个模态的特征进行融合
    def fuse_feats(self, x_rgb: Tuple[Tensor], x_inf: Tuple[Tensor]):
        assert len(x_rgb) == len(x_inf), "Please ensure the len of rgb_feature and inf_feature is same"
        out = []
        for i, (rgb, inf) in enumerate(zip(x_rgb, x_inf)):
            assert rgb.shape == inf.shape, "We need rgb and inf have the same shape for feature of each level"
            out.append(rgb + inf)
        
        return tuple(out)
        
    
    def extract_rgb_feat(self, rgb_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.rgb_backbone(rgb_inputs)
        return x
    
    def extract_inf_feat(self, inf_inputs: Tensor):
        x = self.inf_backbone(inf_inputs)
        return x
    
    def extract_feat(self, batch_inputs: Tensor):
        return super().extract_feat(batch_inputs)