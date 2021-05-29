# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/home/ubuntu/dataset/Cityscape'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'distance_map', 'angle_map']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/trainval',
        ann_dir='gtFine/trainval',
        dt_dir='dt_offset/trainval',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        dt_dir='dt_offset/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/test',
        ann_dir='gtFine/test',
        pipeline=test_pipeline))

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='UNEPT',
    feat_dim=256,
    k=16,
    L=3,
    dropout=0.1,
    heads=8,
    hidden_dim=2048,
    depth=2,
    loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='slide', num_classes=19, stride=(256,256), crop_size=(768, 768), num_queries=9216)

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-8,
                 paramwise_cfg = dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

optimizer_config = dict()
# learning policy
lr_config = dict(policy='step', step=64000, by_epoch=False)
# runtime settings
# total_iters = 640000
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=80000, metric='mIoU')


# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

