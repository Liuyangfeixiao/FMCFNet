import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path
from mmengine.runner import runner
from mmrotate.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets import BaseDetDataset
from mmrotate.utils import register_all_modules

@DATASETS.register_module()
class DroneVehicle(BaseDetDataset):
    """Dataset for DroneVehicle"""
    METAINFO = {
        'classes': ('car', 'freight_car', 'truck', 'bus', 'van'),
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        # self.ann_file 是 IR 模态的数据标签
        with get_local_path(
            self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        # {(1, n) : (0, n-1)}
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"
        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        # TODO need to change data_prefix['img'] to data_prefix['img_path']
        rgb_path = osp.join(self.data_prefix['rgb'], img_info['file_name'])
        inf_path = osp.join(self.data_prefix['inf'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(self.data_prefix['seg'],
                                    img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        
        data_info['rgb_path'] = rgb_path
        data_info['inf_path'] = inf_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            # TODO: 由于图片加入了100 pixel的白边，而 annotation 是按照 (640 * 512) 标注的
            # 所以需要加上 100 的值
            instance['bbox'] = [x + 100 for x in bbox]
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = [[a + 100 for a in x] for x in ann['segmentation']]

            instances.append(instance)
        
        data_info['instances'] = instances
        return data_info

        
# if __name__ == '__main__':
#     register_all_modules()
#     train_pipeline = [
#         dict(type='LoadPairedImageFromFile', backend_args=None),
#         dict(
#             type='mmdet.LoadAnnotations',
#             with_bbox=True,
#             with_mask=True,
#             poly2mask=False),
#         dict(type='ConvertMask2BoxType', box_type='rbox'),
#         dict(type='PairedImageResize', scale=(712, 840), keep_ratio=True),
#         dict(type='PairedImageRandomFlip',
#              prob=0.75,
#              direction=['horizontal', 'vertical', 'diagonal']),
#         dict(type='PairedImagePad', size_divisor=32),
#         dict(type='PackPairedDetInputs',
#              meta_keys=('img_id', 'rgb_path', 'inf_path', 'ori_shape', 'img_shape',
#                         'scale_factor', 'flip', 'flip_direction', 'pad_shape'))
#     ]

#     train_cfg = dict(
#         type='DroneVehicle',
#         data_root='data/DroneVehicle/',
#         ann_file='DV_train_r.json',
#         data_prefix=dict(rgb='train/trainimg', inf='train/trainimgr'),
#         pipeline=train_pipeline,
#         backend_args=None
#     )

#     dataset = DATASETS.build(train_cfg)

#     for item in dataset:
#         print(item)