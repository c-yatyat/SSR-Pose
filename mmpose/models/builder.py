from mmcv.cnn import build_model_from_cfg
from mmcv.utils import Registry

MODELS = Registry('models', build_func=build_model_from_cfg)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
POSENETS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_posenet(cfg):
    """Build posenet."""
    return POSENETS.build(cfg)