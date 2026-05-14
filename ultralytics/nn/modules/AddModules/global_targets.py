import torch

_current_targets = None

def set_current_targets(targets):
    """每个 batch 开始前调用，存储当前 batch 的 targets"""
    global _current_targets
    _current_targets = targets

def get_current_targets():
    """模块内部调用，获取当前 targets"""
    return _current_targets