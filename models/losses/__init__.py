from .gwd_loss import GWDLoss

# 注意：mmdet 内置 loss（如 GaussianFocalLoss、L1Loss）由 mmdet registry 自动提供
# 此处只导出项目自定义的 loss
__all__ = ['GWDLoss']