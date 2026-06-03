# -*- coding: utf-8 -*-
"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""
"""
Main for disentangle (Stage-1) -> CMS (Stage-2, layer4+ only) -> CE (Stage-3) training/eval.
- Keeps CLI/UI parity with prior main_orthogonal.py.
- Ensures lambda_proto / lambda_ortho exist in training_opt.
- Uses training_opt.proto_*_regex for Stage-2 freezing control.

Usage:
  python -m stroketimer.train --cfg <yaml> [--test] [--seed 42] [--trial _SOMENAME]
"""

import os
import argparse
import pprint
import warnings
import random
import inspect
from pathlib import Path

from stroketimer.utils import get_value, resolve_repo_path, source_import

LEGACY_ROOTS = {
    'ImageNet': '/media/Cygnus/haoc/longtail/ImageNet_LT',
    'Places': '/media/Cygnus/haoc/longtail/Places_LT',
    'iNaturalist18': '/media/Cygnus/haoc/longtail/iNaturalist18',
    'CIFAR10': './dataset/CIFAR10',
    'CIFAR100': './dataset/CIFAR100',
}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str, help='Path to YAML config.')
parser.add_argument('--test', default=False, action='store_true', help='Run in evaluation mode.')
parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from YAML.')
parser.add_argument('--test_open', default=False, action='store_true', help='Open-set testing.')
parser.add_argument('--output_logits', default=False, action='store_true', help='Dump logits/labels/paths.')
parser.add_argument('--model_dir', type=str, default=None, help='Override model_dir (folder or final .pth).')
parser.add_argument('--save_feat', type=str, default='', help='Save features for split: train_plain|val|test.')
parser.add_argument('--knn', default=False, action='store_true', help='Enable KNN test path (if available).')
parser.add_argument('--feat_type', type=str, default='cl2n', help='Unused; keep for compatibility.')
parser.add_argument('--dist_type', type=str, default='l2', help='Unused; keep for compatibility.')
parser.add_argument('--val_as_train', default=False, action='store_true', help='Use val as train_val phase.')
parser.add_argument('--dset', type=str, default='CIFAR10', help='Shortcut: dataset name for --exp.')
parser.add_argument('--exp', type=str, default=None, help='Shortcut: config under ./config/<dset>_LT/.')
parser.add_argument('--trial', type=str, default=None, help='Append suffix to log_dir.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--validate-config', action='store_true',
                    help='Validate config paths and exit without loading data or models.')
args = parser.parse_args()

if args.exp:
    args.cfg = f'./config/{args.dset}_LT/{args.exp}.yaml'

def _normalize_config(config):
    for criterion in config.get('criterions', {}).values():
        params = criterion.get('loss_params', {})
        if 'class_freq_path' in params:
            params['class_freq_json'] = params.pop('class_freq_path')
        if 'gamma_freq' in params:
            params['freq_gamma'] = params.pop('gamma_freq')
    return config


def _apply_env_overrides(config, args):
    if os.environ.get('DATA_ROOT'):
        config['training_opt']['data_root'] = os.environ['DATA_ROOT']
    if os.environ.get('CSV_PATH'):
        config.setdefault('ncct', {})['csv_path'] = os.environ['CSV_PATH']
    if os.environ.get('OUTPUT_DIR'):
        config['training_opt']['log_dir'] = os.environ['OUTPUT_DIR']
    if os.environ.get('CLASS_FREQ_JSON'):
        freq_path = os.environ['CLASS_FREQ_JSON']
        for criterion in config.get('criterions', {}).values():
            params = criterion.get('loss_params', {})
            if 'class_freq_json' in params:
                params['class_freq_json'] = freq_path
            if 'class_counts' in params:
                params['class_counts'] = freq_path
    if args.model_dir is None and os.environ.get('CHECKPOINT'):
        args.model_dir = os.environ['CHECKPOINT']
    return config


def _update_from_cli(config, args):
    config = _normalize_config(config)
    config = _apply_env_overrides(config, args)
    config['model_dir'] = get_value(config.get('model_dir'), args.model_dir)
    if 'training_opt' not in config:
        raise KeyError("Config must contain 'training_opt' section.")
    if args.batch_size is not None:
        config['training_opt']['batch_size'] = args.batch_size
    if args.trial:
        config['training_opt']['log_dir'] = config['training_opt']['log_dir'] + f'_{args.trial}'
    # force keys exist
    if 'lambda_proto' not in config['training_opt']:
        config['training_opt']['lambda_proto'] = 0.0
    if 'lambda_ortho' not in config['training_opt']:
        config['training_opt']['lambda_ortho'] = 0.0
    return config


def _validate_config(config):
    errors = []
    for name, block in config.get('networks', {}).items():
        if not block.get('def_file'):
            continue
        p = resolve_repo_path(block.get('def_file', ''))
        if not p.is_file():
            errors.append(f"missing network def_file for {name}: {p}")
    for name, block in config.get('criterions', {}).items():
        if block.get('def_file'):
            p = resolve_repo_path(block.get('def_file', ''))
            if not p.is_file():
                errors.append(f"missing criterion def_file for {name}: {p}")
        params = block.get('loss_params', {})
        for key in ('class_freq_json', 'class_counts'):
            if key in params and isinstance(params[key], str):
                fp = resolve_repo_path(params[key])
                if not fp.is_file():
                    errors.append(f"missing {key} for {name}: {fp}")

    data_root = Path(str(config['training_opt'].get('data_root', '')))
    csv_path = Path(str(config.get('ncct', {}).get('csv_path', '')))
    if not data_root.exists():
        print(f"[validate-config] data_root not found yet (expected for public repo): {data_root}")
    if not csv_path.exists():
        print(f"[validate-config] csv_path not found yet (expected for public repo): {csv_path}")

    if errors:
        for e in errors:
            print(f"[validate-config] ERROR: {e}")
        raise SystemExit(2)
    print("[validate-config] OK: code, loss, model, and class-frequency paths resolve.")


def _strip_yaml_value(value: str) -> str:
    value = value.split("#", 1)[0].strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        value = value[1:-1]
    return value


def _load_validation_config_without_yaml(cfg_path: str) -> dict:
    """Small fallback so --validate-config works before PyYAML is installed."""
    config = {
        "networks": {},
        "criterions": {},
        "training_opt": {},
        "ncct": {},
    }
    section = None
    def_idx = 0
    freq_idx = 0

    with open(cfg_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw.startswith(" ") and stripped.endswith(":"):
                section = stripped[:-1]
                continue

            if stripped.startswith("csv_path:"):
                config["ncct"]["csv_path"] = _strip_yaml_value(stripped.split(":", 1)[1])
            elif stripped.startswith("data_root:"):
                config["training_opt"]["data_root"] = _strip_yaml_value(stripped.split(":", 1)[1])
            elif stripped.startswith("def_file:"):
                target = "criterions" if section == "criterions" else "networks"
                config[target][f"def_file_{def_idx}"] = {
                    "def_file": _strip_yaml_value(stripped.split(":", 1)[1]),
                    "loss_params": {},
                }
                def_idx += 1
            elif stripped.startswith(("class_freq_json:", "class_freq_path:", "class_counts:")):
                key, value = stripped.split(":", 1)
                config["criterions"][f"class_freq_{freq_idx}"] = {
                    "def_file": "",
                    "loss_params": {"class_freq_json" if key == "class_freq_path" else key: _strip_yaml_value(value)},
                }
                freq_idx += 1
    return config


def _load_config(cfg_path: str, validation_only: bool):
    try:
        import yaml
        from yaml import Loader
    except ModuleNotFoundError:
        if validation_only:
            print("[validate-config] PyYAML is not installed; using path-only validation fallback.")
            return _load_validation_config_without_yaml(cfg_path)
        raise RuntimeError("PyYAML is required for training. Install dependencies with: pip install -r requirements.txt")

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=Loader)


assert args.cfg is not None and os.path.isfile(args.cfg), f'cfg not found: {args.cfg}'
config = _load_config(args.cfg, args.validate_config)
config = _update_from_cli(config, args)

if args.validate_config:
    _validate_config(config)
    raise SystemExit(0)

import numpy as np
import torch
from stroketimer.data import dataloader
from stroketimer.runner import model

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_mode = args.test or args.test_open
output_logits = args.output_logits
training_opt = config['training_opt']
dataset = training_opt['dataset']

root_from_cfg = training_opt.get('data_root', None)
fallback_key = dataset.rstrip('_LT')
pretty_root = root_from_cfg if root_from_cfg is not None else LEGACY_ROOTS.get(fallback_key, './')

os.makedirs(training_opt['log_dir'], exist_ok=True)
_effective_cfg_path = os.path.join(training_opt['log_dir'], 'cfg.yaml')
try:
    import yaml
    with open(_effective_cfg_path, 'w', encoding='utf-8') as _f:
        yaml.safe_dump(config, _f, sort_keys=False, allow_unicode=False)
    print(f"==> Saving effective cfg to: {_effective_cfg_path}")
except Exception as _e:
    print(f"[WARN] Failed to save effective cfg: {_e}")

print('Loading dataset from:', pretty_root)
print('==== Effective config (truncated) ==== ')
pprint.pprint(config)


def _split2phase(split_name: str) -> str:
    if split_name == 'train' and args.val_as_train:
        return 'train_val'
    return split_name


def _phase_for_dataset(phase: str, ds: str) -> str:
    """Map runner split name to dataset phase name."""
    if ds == 'ncct_dataset':
        # 'train_plain' and 'train_val' both read the same 'train' samples
        if phase in ('train_plain', 'train_val'):
            return 'train'
    return phase


def _filter_supported_kwargs(func, kwargs: dict) -> dict:
    """Filter kwargs according to dataloader.load_data signature."""
    try:
        sig = inspect.signature(func)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        # Fallback: manual white-list
        safe_keys = {
            'data_root', 'dataset', 'phase', 'batch_size', 'sampler_dic', 'num_workers',
            'shuffle', 'top_k_class', 'cifar_imb_ratio', 'new_classes', 'test_open',
            'ncct_csv_path', 'ncct_transform', 'ncct_fix_depth', 'ncct_zscore',
            'ncct_split_ratios', 'ncct_seed', 'ncct_balance_eval',
        }
        return {k: v for k, v in kwargs.items() if k in safe_keys}


def make_loader(phase: str, batch_size: int, sampler_dic=None, shuffle=True, test_open=False):
    ds_phase = _phase_for_dataset(phase, dataset)

    common_kwargs = dict(
        data_root=training_opt.get('data_root', LEGACY_ROOTS.get(fallback_key, './')),
        dataset=dataset,
        phase=ds_phase,
        batch_size=batch_size,
        sampler_dic=sampler_dic if phase.startswith('train') else None,
        num_workers=training_opt['num_workers'],
        shuffle=shuffle,
    )

    if 'cifar_imb_ratio' in training_opt:
        common_kwargs['cifar_imb_ratio'] = training_opt['cifar_imb_ratio']
    if test_open:
        common_kwargs['test_open'] = True

    if dataset == 'ncct_dataset':
        ncct_cfg = config.get('ncct', {})
        common_kwargs.update(dict(
            ncct_csv_path=ncct_cfg.get('csv_path'),
            ncct_transform=None,
            ncct_fix_depth=ncct_cfg.get('fix_depth', True),
            ncct_zscore=ncct_cfg.get('zscore', True),
            ncct_split_ratios=tuple(ncct_cfg.get('split_ratios', (0.7, 0.15, 0.15))),
            ncct_seed=ncct_cfg.get('seed', args.seed),
            # ★ 控制 val/test 是否做 strict 下采样
            ncct_balance_eval=ncct_cfg.get('balance_eval', True),
        ))

    final_kwargs = _filter_supported_kwargs(dataloader.load_data, common_kwargs)
    missing = set(common_kwargs.keys()) - set(final_kwargs.keys())
    if len(missing) > 0:
        print(f"[main_cms] Skipped unsupported dataloader kwargs: {sorted(list(missing))}")
    return dataloader.load_data(**final_kwargs)


if not test_mode:
    sampler_defs = training_opt.get('sampler', None)
    sampler_dic = None
    if sampler_defs:
        if sampler_defs['type'] == 'ClassAwareSampler':
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
            }
        elif sampler_defs['type'] in ['MixedPrioritizedSampler', 'ClassPrioritySampler']:
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in sampler_defs.items()
                           if k not in ['type', 'def_file']}
            }

    splits = ['train', 'train_plain', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')

    data = {
        sp: make_loader(
            _split2phase(sp),
            batch_size=training_opt['batch_size'],
            sampler_dic=sampler_dic if sp == 'train' else None,
            shuffle=True if sp == 'train' else False
        )
        for sp in splits
    }

    training_model = model(config, data, test=False)
    training_model.train()

else:
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    print('Under testing phase, we still load training data for class-frequency stats.')

    if 'iNaturalist' in training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'
    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    if args.knn or True:
        splits.append('train_plain')

    data = {
        sp: make_loader(
            sp,
            batch_size=training_opt['batch_size'],
            sampler_dic=None,
            shuffle=False,
            test_open=args.test_open
        )
        for sp in splits
    }

    training_model = model(config, data, test=True)
    training_model.load_model(args.model_dir)

    saveit = args.save_feat in ['train_plain', 'val', 'test']
    if saveit:
        test_split = args.save_feat
    training_model.eval(phase=test_split, openset=args.test_open, save_feat=saveit)
    if output_logits and hasattr(training_model, 'output_logits'):
        training_model.output_logits(openset=args.test_open)

print('ALL COMPLETED.')
 
