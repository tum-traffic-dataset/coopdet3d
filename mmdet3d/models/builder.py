from mmcv.utils import Registry

from mmdet.models.builder import BACKBONES, HEADS, LOSSES, NECKS

FUSIONMODELS = Registry("fusion_models")
COOPFUSIONMODELS = Registry("coop_fusion_models")
VTRANSFORMS = Registry("vtransforms")
FUSERS = Registry("fusers")
COOPFUSERS = Registry("coop_fusers")


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_fusion_model(cfg, train_cfg=None, test_cfg=None):
    return FUSIONMODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)

def build_coop_model(cfg, train_cfg=None, test_cfg=None):
    return build_coop_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)


def build_coop_fuser(cfg):
    return COOPFUSERS.build(cfg)


def build_fusion_model_headless(cfg, train_cfg=None, test_cfg=None):
    return FUSIONMODELS.build(cfg)


def build_coop_fusion_model(cfg, train_cfg=None, test_cfg=None):
    return COOPFUSIONMODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )
