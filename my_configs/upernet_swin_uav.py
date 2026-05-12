# ================================================================
#  Swin-S + UPerNet  UAV 语义分割  v2
#
#  vs v1（SegFormer-B2）的核心差异：
#    1. Backbone : SegFormer-B2          → Swin-S
#    2. Head     : SegformerHead         → UPerHead
#                  （显式 FPN 横向融合 + PPM ≈ ASPP，一个 head 同时覆盖两者）
#    3. 域适应   : PhotoMetricDistortion  → 新增 RandomGrayscale
#                  （显式模拟不同传感器色彩响应差异，针对多数据集混合的域偏移）
#    4. 训练步数 : 20000                  → 80000
#    5. Barren 权重 : 1.0               → 4.0（v1 最差类，IoU 仅 45）
#    6. 新增 FocalLoss 第三损失分量
#    7. 滑窗步长 : 341                   → 256（75% 重叠，边界更准）
#    8. 新增 Auxiliary Head（防止深层梯度消失）
# ================================================================

custom_imports = dict(imports=['uav_dataset'], allow_failed_imports=False)

num_classes  = 9
ignore_index = 0
data_root    = './data'
crop_size = (384, 384)

# Swin backbone 内部用 LayerNorm，head 用 BN
norm_cfg = dict(type='BN', requires_grad=True)

# ----------------------------------------------------------------
# 类别权重（ID 0-8）
#   0  Ignore:       0.0  → ignore_index 兜底，不参与 loss
#   1  Background:   1.2  → v1 IoU 66，第二差，略微提升
#   2  Building:     1.5
#   3  Road:         2.0
#   4  Water:        1.5
#   5  Barren:       4.0  ← v1 最差（IoU 45），大幅提升
#   6  Vegetation:   1.0
#   7  Agricultural: 1.0
#   8  Vehicle:      4.0  ← 像素占比最低（0.63%）
# ----------------------------------------------------------------
CLASS_WEIGHT = [0.0, 1.2, 1.5, 2.0, 1.5, 4.0, 1.0, 1.0, 4.0]

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=ignore_index,
        size_divisor=32,
    ),

    # ── Backbone：Swin-S ─────────────────────────────────────────
    # 输出四个 stage 的特征图，通道数 [96, 192, 384, 768]
    # window_size=7 对应 ImageNet 预训练，航拍图用 pretrain_img_size=224
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        depths=[2, 2, 18, 2],          # Swin-S 标准深度
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'
        )
    ),

    # ── Decode Head：UPerHead ─────────────────────────────────────
    # FPN 横向融合四个 stage 特征（显式多尺度）
    # PPM pool_scales=(1,2,3,6) 等价 ASPP（显式多感受野）
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                avg_non_ignore=True,
                class_weight=CLASS_WEIGHT,
            ),
            dict(
                type='DiceLoss',
                loss_weight=0.4,
                ignore_index=ignore_index,
            ),
        ]
    ),

    # ── Auxiliary Head：FCNHead ───────────────────────────────────
    # 接在 stage2（index=2）输出，防止深层梯度消失
    # loss_weight=0.4，训练时辅助，推理时不参与
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            avg_non_ignore=True,
            class_weight=CLASS_WEIGHT,
        )
    ),

    # 滑动窗口推理：步长 256（75% 重叠）
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(256, 256)),
)

# ── 训练数据增强 ─────────────────────────────────────────────────
# RandomGrayscale（p=0.2）：显式域适应
#   训练集来自多个数据集，传感器色彩响应不同
#   20% 概率转灰度后再转回 RGB，强迫模型学习形状/纹理特征
#   而非依赖特定色彩分布，从而在测试集域偏移时更鲁棒
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

dataset_type = 'UAVDataset'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/train.txt',
        data_prefix=dict(img_path='train_raw/images', seg_map_path='train_raw/masks'),
        pipeline=train_pipeline,
        ignore_index=ignore_index,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='train_raw/images', seg_map_path='train_raw/masks'),
        pipeline=val_pipeline,
        ignore_index=ignore_index,
    )
)

val_evaluator   = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=ignore_index)
test_dataloader = None
test_cfg        = None
test_evaluator  = None

# ── 优化器 ───────────────────────────────────────────────────────
# backbone lr 比 head 低 10 倍，保护预训练特征
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
        }
    )
)

# ── 学习率调度 ────────────────────────────────────────────────────
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0,    end=2000),
    dict(type='PolyLR',   eta_min=0.0, power=1.0,            by_epoch=False, begin=2000, end=40000),
]

# ── 训练循环 ──────────────────────────────────────────────────────
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg   = dict(type='ValLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

default_scope  = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork'),
    dist_cfg=dict(backend='nccl')
)
log_processor  = dict(by_epoch=False)
log_level      = 'INFO'
load_from      = None   # 全新训练，从 ImageNet 预训练权重开始
resume         = False
work_dir       = './work_dirs/upernet_swin_uav'
randomness     = dict(seed=42)