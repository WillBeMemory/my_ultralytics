# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics neural network modules.

This module provides access to various neural network components used in Ultralytics models, including convolution
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron
    >>> from ultralytics.nn.modules import Conv
    >>> import torch
    >>> import subprocess
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # pip install onnxslim
"""
# from .backbone.lsknet import LSKNet
# from .IndexSelector import IndexSelector
from .seNet import SEAttention
from .CPFNet import EVCBlock
from ultralytics.nn.modules.hwd import HWD
from .lsknet import LSK,C3k2_LSK
from ultralytics.nn.modules.fmb import FMB
from ultralytics.nn.modules.simam import SimAM,C3k2_SimAM
from ultralytics.nn.modules.coordatt import CoordAtt, C3k2_CA
from ultralytics.nn.modules.WTConv import WTConv2d,C3k2_WT
from ultralytics.nn.modules.shsa import SHSA

from ultralytics.nn.modules.mobilemqa import A2C2f_MobileMQA
from ultralytics.nn.modules.WeightedP2Fusion import WeightedP2Fusion
from ultralytics.nn.modules.SCSA import A2C2f_SCSA,C2PSA_SCSA
from ultralytics.nn.modules.CSPPConv import CSP_PConv
from ultralytics.nn.modules.AdaptiveResidualFusion import AdaptiveResidualFusion
from ultralytics.nn.modules.spapf import SPAPF
from ultralytics.nn.modules.DyHead import DyHead
from ultralytics.nn.modules.SPPF_LSKA import SPPF_LSKA
from ultralytics.nn.modules.DynamicMultiBranch import DynamicPartialIdentity,DynamicWaveletAttentionIdentity,DynamicWaveletIdentity
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d
from ultralytics.nn.modules.DyUpsample import DySample
from ultralytics.nn.modules.MoE import MoEBlock,AttMoE,ConvMoE
from ultralytics.nn.modules.DySPPF import DynamicSPPF
from ultralytics.nn.modules.RFAConv import RFAConv
from ultralytics.nn.modules.DPConv import DPConv
from ultralytics.nn.modules.TernaryDPConv import TernaryDPConv
from ultralytics.nn.modules.SmartAreaAttention import SmartAreaAttention
from ultralytics.nn.modules.SaveFirstImage import SaveFirstImage
from ultralytics.nn.modules.BiDirectionalTGFI import BiDirectionalTGFI,BiDirectionalTGFIBlock
from ultralytics.nn.modules.AddModules.BackgroundSuppression import BackgroundSuppression
from ultralytics.nn.modules.TernaryMoEBlock import TernaryMoEBlock
from ultralytics.nn.modules.DeterministicGateConv import DeterministicGateConv
from ultralytics.nn.modules.GatedC3k2 import GatedC3k2
from ultralytics.nn.modules.WTGatedC3k2 import WTGatedC3k2
from ultralytics.nn.modules.HIPA import HIPA
from ultralytics.nn.modules.HIPAV2 import HIPAV2
from ultralytics.nn.modules.WaveletStem import WaveletStem
from ultralytics.nn.modules.DSWT_GhostConv import DSWT_GhostConv
from ultralytics.nn.modules.DynamicC3k2 import DynamicC3k2
from ultralytics.nn.modules.CARAFE import CARAFE
from ultralytics.nn.modules.BiFPN import BiFPN
from ultralytics.nn.modules.SplitList import SplitList
from ultralytics.nn.modules.CCFPN import CCFPN
from ultralytics.nn.modules.WaveletCSP import WaveletCSP
from ultralytics.nn.modules.WaveletRefine import WaveletRefine

from ultralytics.nn.modules.SlimDetect import SlimDetect
from ultralytics.nn.modules.IWConv import IWConv
from ultralytics.nn.modules.SPDConv import SPDConv
from ultralytics.nn.modules.InterpDownsample import InterpDownsample
from ultralytics.nn.modules.ChannelGateSepConv import ChannelGateSepConv
from ultralytics.nn.modules.ChannelGateSepConv import ChannelGateSepConv


from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import (
    OBB,
    OBB26,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    Pose26,
    RTDETRDecoder,
    Segment,
    Segment26,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
    v10Detect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "AIFI",
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CBAM",
    "CIB",
    "DFL",
    "ELAN1",
    "MLP",
    "OBB",
    "OBB26",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "A2C2f",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ChannelAttention",
    "Classify",
    "Concat",
    "ContrastiveHead",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "Detect",
    "Focus",
    "GhostBottleneck",
    "GhostConv",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Index",
    "LRPCHead",
    "LayerNorm2d",
    "LightConv",
    "MLPBlock",
    "MSDeformAttn",
    "MaxSigmoidAttnBlock",
    "Pose",
    "Pose26",
    "Proto",
    "RTDETRDecoder",
    "RepC3",
    "RepConv",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "Segment",
    "Segment26",
    "SpatialAttention",
    "TorchVision",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "YOLOESegment26",
    "v10Detect",
)
