import os
import yaml
import inspect
import importlib
import torchvision
torchvision.disable_beta_transforms_warning()

GLOBAL_CONFIG = dict()

def register(cls: type):
    if cls.__name__ in GLOBAL_CONFIG:
        raise ValueError('{} already registered'.format(cls.__name__))

    if inspect.isfunction(cls):
        GLOBAL_CONFIG[cls.__name__] = cls

    elif inspect.isclass(cls):
        GLOBAL_CONFIG[cls.__name__] = cls

    else:
        raise ValueError(f'register {cls}')

    return cls

def create(cfg: dict, **kwargs):
    cfg = cfg.copy()
    cfg.update(kwargs)
    keys = list(cfg.keys())

    if 'type' in keys:
        name = cfg['type']
        cfg['type'] = GLOBAL_CONFIG[name]

    keys.remove('type')
    for k in keys:
        if isinstance(cfg[k], dict) and 'type' in cfg[k].keys():
            cfg[k] = create(cfg[k])
        elif isinstance(cfg[k], list):
            cfg[k] = [create(v) if isinstance(v, dict) else v for v in cfg[k]]

    cls = cfg.pop('type', None)
    if inspect.isfunction(cls):
        return cls


    cfg['type'] = cls(**cfg)
    return cfg['type']