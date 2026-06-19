"""Deprecated backward-compat shim. 规范模块为 ultralytics.nn.modules.FPN_PAN_WRF。

本文件**仅为**让旧 .pt 权重可加载而存在:旧 checkpoint 是用 torch.load() 反序列化的,
pickle 按类的原始模块路径 ``ultralytics.nn.modules.FPN_PAN_BiFPN`` 查找类,故该路径必须仍可 import。
新代码一律从 FPN_PAN_WRF 导入并使用 WRF 命名,勿用此处的旧名。

彻底移除本 shim 的办法:用本 shim 先把每个旧 .pt 加载进来,再重新保存一次——
重新保存的 checkpoint 将引用新的 FPN_PAN_WRF 路径,届时可删除本文件。
"""
from ultralytics.nn.modules.FPN_PAN_WRF import (  # noqa: F401  (re-export for pickle find_class)
    FPN_PAN,
    DepthwiseSeparableConv,
    WeightedAdd,
    WRF,
    FPN_PAN_WRF,
    FPN_PAN_BiFPN,  # = FPN_PAN_WRF（别名，定义在 FPN_PAN_WRF 内）
)

# 旧 .pt 以这些旧类名 + 本模块路径 pickle 的，按旧名重新暴露：
BiFPNLayer = WRF           # noqa: F401
BiFPN_Add = WeightedAdd    # noqa: F401
