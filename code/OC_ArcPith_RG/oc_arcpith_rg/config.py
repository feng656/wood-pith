from pathlib import Path
import yaml

def load_config(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['_root'] = str(path.resolve().parent)
    return cfg

def resolve_path(cfg, p):
    p = Path(p)
    return p if p.is_absolute() else Path(cfg['_root']) / p
