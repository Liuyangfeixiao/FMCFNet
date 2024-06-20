_base_ = ['../_base_/default_runtime.py', '../_base_/schedules/schedule_6x.py',
          './dronevehicle.py']
max_epochs = 50
base_lr = 0.004 / 16
interval = 1

angle_version = 'le90'

model = dict(
    type='FMCFNet_Detector',
    data_preprocessor = dict(
        type='PairedDetDataPreprocessor',
        mean_rgb=[159.25, 159.97, 160.34],
        std_rgb=[91.36, 90.87, 90.90],
        mean_inf=[186.75, 186.75, 186.75],
        std_inf=[72.58, 72.58, 72.58],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_value=255,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        type='FMCFNet',
        dims_in=[512, 1024, 2048],
        sr_ratio=[3, 2, 1],
        n_layers=[4, 4, 4],
        num_heads=[8, 8, 8],
        vert_ahchors=[40, 20, 10],
        horz_anchors=[40, 20, 10],
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg = dict(type='Pretrained', checkpoint='pretrained_weights/resnet50-2stream.pth')
        ),
    neck=dict(
        type='mmdet.CSPNeXtPAFPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='RotatedRTMDetSepBNHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        angle_version=angle_version,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',
            iou_calculator=dict(type='RBboxOverlaps2D'),
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=500,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.2),
        max_per_img=200),
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)