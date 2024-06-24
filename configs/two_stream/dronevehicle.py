# dataset settings
dataset_type = 'DroneVehicle'
data_root = 'data/DroneVehicle'
backend_args = None

train_pipeline = [
        dict(type='LoadPairedImageFromFile', backend_args=backend_args),
        dict(
            type='mmdet.LoadAnnotations',
            with_bbox=True,
            with_mask=True,
            poly2mask=False),
        dict(type='ConvertMask2BoxType', box_type='rbox'),
        # (712, 840)
        dict(type='PairedImageResize', scale=(640, 640), keep_ratio=True),
        dict(type='PairedImageRandomFlip',
             prob=0.75,
             direction=['horizontal', 'vertical', 'diagonal']),
        # dict(type='PairedImagePad', size_divisor=32),
        dict(type='PackPairedDetInputs',
             meta_keys=('img_id', 'rgb_path', 'inf_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction'))
    ]

val_pipeline = [
    dict(type='LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='PairedImageResize', scale=(640, 640), keep_ratio=True),
    dict( # avoid bboxes being resized
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ConvertMask2BoxType', box_type='rbox'),
    dict(type='PackPairedDetInputs',
        meta_keys=('img_id', 'rgb_path', 'inf_path', 'ori_shape', 'img_shape',
                'scale_factor', 'instances'))
]

test_pipeline = [
    dict(type='LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='PairedImageResize', scale=(640, 640), keep_ratio=True),
    dict(type='PackPairedDetInputs',
        meta_keys=('img_id', 'rgb_path', 'inf_path', 'ori_shape', 'img_shape',
                'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='DV_train_r.json',
        data_prefix=dict(rgb='train/trainimg', inf='train/trainimgr'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='DV_test_r.json',
        data_prefix=dict(rgb='test/testimg', inf='test/testimgr'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='DV_test_r.json',
        data_prefix=dict(rgb='test/testimg', inf='test/testimgr'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(type='DOTAMetric', metric='mAP')
# val_evaluator = dict(
#     type='RotatedCocoMetric',
#     metric='bbox',
#     classwise=True,
#     backend_args=backend_args)

test_evaluator = val_evaluator