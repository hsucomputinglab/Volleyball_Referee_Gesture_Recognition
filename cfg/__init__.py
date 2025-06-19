from types import SimpleNamespace
from pathlib import Path
import yaml
import os


def load_config(file_path='data.yaml', append_filename=False):
    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)
        file_cfg = file_cfg or {}  # empty file returns None
    return file_cfg


def check_cfg(cfg, global_values):
    for k, v in global_values.items():
        key_search(cfg, k, v)
    return cfg


def key_search(cfg, k, v):
    for key, value in cfg.items():
        if isinstance(value, dict):
            key_search(value, k, v)
        if key == k and v is not None:
            cfg[key] = v




def cfg2dict(file_path):
    yaml_cfg = dict()
    if isinstance(file_path, (str, Path)):
        cfg = load_config(file_path)
    elif isinstance(file_path, SimpleNamespace):
        cfg = vars(file_path)

    if '__include__' in cfg:
        for include in cfg['__include__']:
            config = load_config(os.path.join(os.path.dirname(file_path), include))
            yaml_cfg.update(config)
        del cfg['__include__']

    return yaml_cfg, cfg


def update_cfg(file_path, args):
    config, model = cfg2dict(file_path)
    model.update(args)

    for k, v in model.items():
        if k in config.keys():
            model[k] = config[k] if v is None else v
        else:
            model[k] = v
    config.update(model)

    global_values = {k: v for k, v in config.items() if not isinstance(v, (dict, list))}
    global_dict = {k: v for k, v in model.items() if isinstance(v, dict)}
    global_list = {k: v for k, v in model.items() if isinstance(v, list)}

    config = check_cfg(config, global_dict)
    config = check_cfg(config, global_list)
    config = check_cfg(config, global_values)

    # for k, v in global_dict.items():
    #     del config[k]
    # for k, v in global_list.items():
    #     del config[k]

    return config
