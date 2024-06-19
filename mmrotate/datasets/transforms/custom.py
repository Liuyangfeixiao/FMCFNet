import cv2
import mmcv
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadImageFromFile
from mmcv.transforms.utils import cache_randomness
from mmengine.structures import InstanceData, PixelData
from mmdet.structures.bbox import BaseBoxes, get_box_type, autocast_box_type
from mmdet.structures.mask import PolygonMasks
from mmdet.structures import DetDataSample
from mmdet.datasets.transforms import Resize, RandomFlip, PackDetInputs, Pad

from mmengine.utils import is_list_of
import mmengine.fileio as fileio
from mmrotate.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadPairedImageFromNDArray(LoadImageFromFile):
    """Load rgb and inf image from ndarray

    Required Keys:
    - rgb_img
    - inf_img

    Modified Keys:
    - rgb_img
    - inf_img
    - img_shape
    - ori_shape
    """
    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information
        Args:
            results: contains `rgb_img` and `inf_img`
        Returns:
            dict: The dict contains loaded images and meta information.
        """
        rgb_img = results['rgb_img']
        inf_img = results['inf_img']
        if self.to_float32:
            rgb_img = rgb_img.astype(np.float32)
            inf_img = inf_img.astype(np.float32)
        results['rgb_path'] = None
        results['inf_path'] = None
        results['inf_img'] = inf_img
        results['rgb_img'] = rgb_img
        results['img_shape'] = rgb_img.shape[:2]
        results['ori_shape'] = rgb_img.shape[:2]
        return results

@TRANSFORMS.register_module()
class LoadPairedImageFromFile(LoadImageFromFile):
    """ Load rgb and inf image from file
    
    Required Keys:
    - rgb_path
    - inf_path

    Modified Keys:
    - rgb_img
    - inf_img
    - img_shape
    - ori_shape

    """
    def transform(self, results: dict):
        """Functions to load RGB and Infracture image
        
        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """
        rgb_filename = results['rgb_path']
        inf_filename = results['inf_path']

        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, rgb_filename)
                rgb_bytes = file_client.get(rgb_filename)
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, inf_filename)
                inf_bytes = file_client.get(inf_filename)
            else:
                rgb_bytes = fileio.get(
                    rgb_filename, backend_args=self.backend_args)
                inf_bytes = fileio.get(
                    inf_filename, backend_args=self.backend_args)
            
            rgb_img = mmcv.imfrombytes(
                rgb_bytes, flag=self.color_type, backend=self.imdecode_backend)
            inf_img = mmcv.imfrombytes(
                inf_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        
        assert rgb_img is not None, f"failed to load rgb imgae: {rgb_filename}"
        assert inf_img is not None, f"failed to load inf image: {inf_filename}"

        if self.to_float32:
            rgb_img = rgb_img.astype(np.float32)
            inf_img = inf_img.astype(np.float32)
        
        results['rgb'] = rgb_img
        results['inf'] = inf_img
        results['img_shape'] = rgb_img.shape[:2]
        results['ori_shape'] = rgb_img.shape[:2]

        return results
    
@TRANSFORMS.register_module()
class PairedImageResize(Resize):
    def _resize_img(self, results: dict) -> None:
        if results.get('rgb', None) is not None:
            if self.keep_ratio:
                rgb_img, scale_factor = mmcv.imrescale(
                    results['rgb'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)

                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = rgb_img.shape[:2]
                h, w = results['rgb'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                rgb_img, w_scale, h_scale = mmcv.imresize(
                    results['rgb'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['rgb'] = rgb_img
            results['img_shape'] = rgb_img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio
        
        if results.get('inf', None) is not None:
            if self.keep_ratio:
                inf_img, scale_factor = mmcv.imrescale(
                    results['inf'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)

                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = inf_img.shape[:2]
                h, w = results['inf'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                inf_img, w_scale, h_scale = mmcv.imresize(
                    results['inf'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['inf'] = inf_img

@TRANSFORMS.register_module()
class PairedImageRandomFlip(RandomFlip):
    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the RandomFlip."""
        cur_dir = results['flip_direction']
        h, w = results['rgb'].shape[:2]

        if cur_dir == 'horizontal':
            homography_matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'vertical':
            homography_matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'diagonal':
            homography_matrix = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        else:
            homography_matrix = np.eye(3, dtype=np.float32)

        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']
    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['rgb'] = mmcv.imflip(
            results['rgb'], direction=results['flip_direction'])
        results['inf'] = mmcv.imflip(
            results['inf'], direction=results['flip_direction'])

        img_shape = results['rgb'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])
            
        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])
        
        # record homography matrix for flip
        self._record_homography_matrix(results)

@TRANSFORMS.register_module()
class PairedImagePad(Pad):
    def _pad_img(self, results: dict) -> None:
        """Pad Images according to ``self.size``
        """
        # padding 的值
        pad_val = self.pad_val.get('img', 0)

        size = None  # (h, w)
        if self.pad_to_square:
            max_size = max(results['rgb'].shape[:2])
            size = (max_size, max_size)
        if self.size_divisor is not None:
            if size is None:
                size = (results['rgb'].shape[0], results['rgb'].shape[1])
            pad_h = int(np.ceil(
                size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(
                size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size[::-1]
        if isinstance(pad_val, int) and results['rgb'].ndim == 3:
            pad_val = tuple(pad_val for _ in range(results['rgb'].shape[2]))
        
        padded_rgb = mmcv.impad(
            results['rgb'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)
        
        if isinstance(pad_val, int) and results['inf'].ndim == 3:
            pad_val = tuple(pad_val for _ in range(results['inf'].shape[2]))
        
        padded_inf = mmcv.impad(
            results['inf'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)
        
        results['rgb'] = padded_rgb
        results['inf'] = padded_inf
        results['pad_shape'] = padded_rgb.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        results['img_shape'] = padded_rgb.shape[:2]


@TRANSFORMS.register_module()
class PackPairedDetInputs(PackDetInputs):
    def __init__(self, meta_keys=('img_id', 'rgb_path', 'inf_path', 'ori_shape', 'img_shape',
                                  'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys
    
    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'rgb' (obj:`torch.Tensor` or 'dict'): The forward rgb data of models. (C, H, W)
            - 'inf' (obj:`torch.Tensor` or 'dict'): The forward inf data of models. (C, H, W)
            - 'data_samples' (obj:`DetDataSample`): The annotation info of the sample.
        """
        packed_results = dict()
        if 'rgb' in results:
            rgb = results['rgb']
            if len(rgb.shape) < 3:
                rgb = np.expand_dims(rgb, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not rgb.flags.c_contiguous:
                rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1))
                rgb = to_tensor(rgb)
            else:
                rgb = to_tensor(rgb).permute(2, 0, 1).contiguous()

            packed_results['rgb'] = rgb
        
        if 'inf' in results:
            inf = results['inf']
            if len(inf.shape) < 3:
                inf = np.expand_dims(inf, -1)
            if not inf.flags.c_contiguous:
                inf = np.ascontiguousarray(inf.transpose(2, 0, 1))
                inf = to_tensor(inf)
            else:
                inf = to_tensor(inf).permute(2, 0, 1).contiguous()

            packed_results['inf'] = inf
        
        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]
        
        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]
        
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

