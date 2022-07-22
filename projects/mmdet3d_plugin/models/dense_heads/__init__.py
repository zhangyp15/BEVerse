# Copyright (c) OpenMMLab. All rights reserved.
from .map_head import MapHead
from .mtl_head import MultiTaskHead
from .motion_head import MotionHead
from .det_head import CenterHeadv1

__all__ = ['MapHead', 'MultiTaskHead', 'MotionHead', 'CenterHeadv1']
