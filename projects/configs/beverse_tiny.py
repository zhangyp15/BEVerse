_base_ = [
    './beverse_singleframe_tiny.py'
]

receptive_field = 3
future_frames = 4
future_discount = 0.95

model = dict(
    temporal_model=dict(
        type='Temporal3DConvModel',
        receptive_field=receptive_field,
        input_egopose=True,
        in_channels=64,
        input_shape=(128, 128),
        with_skip_connect=True,
    ),
    pts_bbox_head=dict(
        task_enbale={
            '3dod': True, 'map': True, 'motion': True,
        },
        task_weights={
            '3dod': 1.0, 'map': 10.0, 'motion': 1.0,
        },
        cfg_motion=dict(
            type='IterativeFlow',
            task_dict={
                'segmentation': 2,
                'instance_center': 1,
                'instance_offset': 2,
                'instance_flow': 2,
            },
            receptive_field=receptive_field,
            n_future=future_frames,
            using_spatial_prob=True,
            n_gru_blocks=1,
            future_discount=future_discount,
            loss_weights={
                'loss_motion_seg': 5.0,
                'loss_motion_centerness': 1.0,
                'loss_motion_offset': 1.0,
                'loss_motion_flow': 1.0,
                'loss_motion_prob': 10.0,
            },
            sample_ignore_mode='past_valid',
            posterior_with_label=False,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ),
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            receptive_field=receptive_field,
            future_frames=future_frames,
        ),
    ),
    val=dict(
        receptive_field=receptive_field,
        future_frames=future_frames,
    ),
    test=dict(
        receptive_field=receptive_field,
        future_frames=future_frames,
    ),
)
