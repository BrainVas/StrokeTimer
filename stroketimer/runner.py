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
Runner with 3-stage schedule using CMS (Contrastive Mean-Shift) in Stage-2.
- Stage-1: disentanglement (reconstruction L1, KL, z-reencode, orthogonality).
- Stage-2: CMSLoss on L2-normalized semantic features (no projector).
- Stage-3: CE (e.g., Balanced Softmax) on classifier with optional linear ramp.

Key policy (your request):
- S1/S2: classifier is frozen (no gradients), but we STILL compute logits in batch_forward()
         under no_grad() for logging/evaluation consistency.
- S3: the whole encoder is frozen; classifier is unfrozen and trained with CE.
      In batch_loss(), we recompute classifier logits WITH grad for CE,
      and then overwrite self.logits with a detached copy for metrics.

Compatibility:
- Expects feat_model forward signature:
      forward(x, cls_only: bool=False) -> (semantic_feat, featmap or None)
  And caches (for Stage-1) on feat_model.module or feat_model:
      cache_x_hat, cache_mu, cache_logvar, cache_z, cache_s_vec
      reencode_z()  (optional)
- Classifier adapter supports both:
      logits, _ = classifier(features, centroids, phase, labels)
  OR
      logits = classifier(features)
"""

import os
import copy
import math
import pickle
import json
import sys
import time
import warnings
import inspect
import re
from typing import Any, Tuple, List, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# headless plotting
import matplotlib
matplotlib.use("Agg")
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from stroketimer.utils import *   # print_write, mic_acc_cal, shot_acc, weighted_mic_acc_cal, weighted_shot_acc, class_count, torch2numpy, CosineAnnealingLRWarmup, get_priority
from stroketimer.logger import Logger


def source_import(file_path: str):
    """Import a python file by path and return the loaded module."""
    from stroketimer.utils import source_import as _source_import
    return _source_import(file_path)


class HiddenPrints:
    """Silence stdout within a with-context."""
    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, a, b, c):
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = self._orig


class model:
    """Main runner (train/eval/save) with CMS Stage-2."""

    def __init__(self, config, data, test=False, meta_sample=False, learner=None):
        self.meta_sample = meta_sample
        if meta_sample:
            assert learner is not None
            self.learner = learner
            self.meta_data = iter(data['meta'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = config['training_opt']
        self.memory = config.get('memory', {})
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config.get('shuffle', False)

        # ---- init state ----
        self.centers = None
        self.centroids = None
        self.eval_acc_mic_top1 = 0.0
        self.many_acc_top1 = 0.0
        self.median_acc_top1 = 0.0
        self.low_acc_top1 = 0.0

        # disentanglement weights (Stage-1)
        self.lambda_rec   = float(self.training_opt.get('lambda_rec',   0.0))
        self.lambda_kl    = float(self.training_opt.get('lambda_kl',    0.0))
        self.lambda_zrec  = float(self.training_opt.get('lambda_zrec',  0.0))
        self.lambda_ortho = float(self.training_opt.get('lambda_ortho', 0.0))
        self.ortho_use_mu = bool(self.training_opt.get('ortho_use_mu', True))

        # 3-stage schedule
        self.disent_epochs  = int(self.training_opt.get('disent_epochs', 0))
        self.proto_start    = int(self.training_opt.get('proto_start', self.disent_epochs + 1))
        self.cls_start      = int(self.training_opt.get('cls_start',   self.disent_epochs + 1))
        self.freeze_low_at_proto = bool(self.training_opt.get('freeze_low_layers_at_proto', True))
        self.disable_disent_after_proto = bool(self.training_opt.get('disable_disent_after_proto', True))
        self.freeze_encoder_at_cls = bool(self.training_opt.get('freeze_encoder_at_cls', True))

        # Stage-2/3 loss multipliers
        self.lambda_proto  = float(self.training_opt.get('lambda_proto', 0.0))  # CMS in Stage-2
        self.lambda_ce     = float(self.training_opt.get('lambda_ce',    1.0))  # CE in Stage-3

        # Stage-3 CE weight ramp
        self.ce_ramp = int(self.training_opt.get('ce_ramp', 0))
        self.ce_end_weight = float(self.training_opt.get('ce_end_weight', 1.0))
        self.ce_weight_current = 0.0

        # Regex control of freezing in Stage-2
        self.proto_freeze_regex = self.training_opt.get('proto_freeze_regex', [])
        self.proto_unfreeze_regex = self.training_opt.get('proto_unfreeze_regex', [])

        # Epoch bridging (if iterations are given)
        if self.training_opt.get('num_iterations'):
            steps_per_epoch = max(1, len(self.data['train']))
            self.training_opt['num_epochs'] = math.ceil(self.training_opt['num_iterations'] / steps_per_epoch)
        if self.config.get('warmup_iterations'):
            steps_per_epoch = max(1, len(self.data['train']))
            self.config['warmup_epochs'] = math.ceil(self.config['warmup_iterations'] / steps_per_epoch)

        os.makedirs(self.training_opt['log_dir'], exist_ok=True)
        self.logger = Logger(self.training_opt['log_dir'])

        self._stage2_frozen_applied = False
        self._stage3_frozen_applied = False

        # build models, optimizers, losses
        self.init_models()
        if self.config.get('model_dir') is not None:
            self.load_model(self.config['model_dir'])

        if not self.test_mode:
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = max(1, int(self.training_data_num / self.training_opt['batch_size']))

            self.scheduler_params = self.training_opt.get('scheduler_params', None)
            self.model_optimizer, self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            self.init_criterions()

            if self.memory.get('init_centroids', False) and 'FeatureLoss' in self.criterions:
                # If you need class prototypes for other losses, compute once:
                self.criterions['FeatureLoss'].centroids = self.centroids_cal(self.data['train_plain'])

            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            if os.path.isfile(self.log_file):
                try: os.remove(self.log_file)
                except Exception: pass
            self.logger.log_cfg(self.config)
        else:
            self.log_file = None

    # ------------------------------------------------------------------------------------------
    # Build networks and default optimizer param groups
    # ------------------------------------------------------------------------------------------
    def init_models(self, optimizer=True):
        networks_defs = self.config['networks']
        self.networks: Dict[str, nn.Module] = {}
        self.model_optim_params_list: List[dict] = []

        print("Using", torch.cuda.device_count(), "GPUs.")
        for key, val in networks_defs.items():
            create_fn = source_import(val['def_file']).create_model
            model_args = dict(val['params'])

            # Keep only supported kwargs
            try:
                sig = inspect.signature(create_fn)
                allowed = set(sig.parameters.keys())
                clean_args = {k: v for k, v in model_args.items() if k in allowed}
            except (TypeError, ValueError):
                clean_args = model_args

            self.networks[key] = create_fn(**clean_args)
            # DP wrap unless special classifier
            if 'KNNClassifier' in type(self.networks[key]).__name__:
                self.networks[key] = self.networks[key].to(self.device)
            else:
                self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)

            # Optional global freeze
            if val.get('fix', False):
                print('Global freeze on feature weights except "selfatt"/"fc" (if exist).')
                for n, p in self.networks[key].named_parameters():
                    if ('selfatt' not in n) and ('fc' not in n):
                        p.requires_grad = False

            # Diagnostic print
            for n, p in self.networks[key].named_parameters():
                print(n, p.requires_grad)

        self.current_epoch = 1

        # Collect default param groups (only trainable params)
        for key, val in self.config['networks'].items():
            op = val.get('optim_params', None)
            if op is None:
                continue
            trainable = [p for p in self.networks[key].parameters() if p.requires_grad]
            if len(trainable) == 0:
                continue
            self.model_optim_params_list.append({
                'params': trainable,
                'lr': op.get('lr', 0.001),
                'momentum': op.get('momentum', 0.9),
                'weight_decay': op.get('weight_decay', 0.0),
            })

        # Stage-1 starts with classifier frozen
        if not self.test_mode:
            self._set_classifier_trainable(False)
            self._rebuild_optimizer_with_current_trainable()

    # ------------------------------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------------------------------
    def init_criterions(self):
        criterion_defs = self.config.get('criterions', {})
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            loss_fn = source_import(val['def_file']).create_loss
            self.criterions[key] = loss_fn(**val['loss_params']).to(self.device)
            self.criterion_weights[key] = float(val.get('weight', 1.0))

            if val.get('optim_params'):
                o = val['optim_params']
                opt_params = [{
                    'params': [p for p in self.criterions[key].parameters() if p.requires_grad],
                    'lr': o['lr'],
                    'momentum': o.get('momentum', 0.9),
                    'weight_decay': o.get('weight_decay', 0.0),
                }]
                self.criterion_optimizer, self.criterion_optimizer_scheduler = self.init_optimizers(opt_params)
            else:
                self.criterion_optimizer = None
                self.criterion_optimizer_scheduler = None

    def init_optimizers(self, optim_params: List[dict]):
        optim_params = [pg for pg in optim_params if pg.get('params')]
        if len(optim_params) == 0:
            # dummy optimizer
            dummy = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
            optimizer = optim.SGD([{'params': [dummy], 'lr': 0.0}])
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
            return optimizer, scheduler

        optimizer = optim.SGD(optim_params)
        if self.config.get('coslr', False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt['num_epochs'], eta_min=self.config.get('endlr', 0.0))
        elif self.config.get('coslrwarmup', False):
            scheduler = CosineAnnealingLRWarmup(
                optimizer=optimizer,
                T_max=self.training_opt['num_epochs'],
                eta_min=self.config.get('endlr', 0.0),
                warmup_epochs=self.config.get('warmup_epochs', 0),
                base_lr=self.config.get('base_lr', 0.0),
                warmup_lr=self.config.get('warmup_lr', 0.0)
            )
        elif self.scheduler_params:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.scheduler_params['step_size'],
                gamma=self.scheduler_params['gamma'])
        else:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        return optimizer, scheduler

    # ------------------------------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------------------------------
    def _set_classifier_trainable(self, trainable: bool):
        clf = self.networks.get('classifier', None)
        if clf is None:
            return
        for p in clf.parameters():
            p.requires_grad = trainable
        print(f"[Freeze] Classifier trainable = {trainable}")

    def _apply_regex_freeze(self, module: nn.Module, freeze_list: List[str], unfreeze_list: List[str]):
        """
        Stage-2: freeze everything matched by freeze_list, then force-unfreeze items matched by unfreeze_list.
        Assumes DataParallel wrapper; strip optional 'module.' prefix for regex.
        """
        def _strip(name: str) -> str:
            return name[7:] if name.startswith("module.") else name

        # freeze by blacklist
        if freeze_list:
            for n, p in module.named_parameters():
                base = _strip(n)
                if any(re.search(r, base) for r in freeze_list):
                    p.requires_grad = False

        # force-unfreeze by whitelist
        if unfreeze_list:
            for n, p in module.named_parameters():
                base = _strip(n)
                if any(re.search(r, base) for r in unfreeze_list):
                    p.requires_grad = True

        # print summary
        print("[Stage-2 Freeze Summary] trainable params:")
        for n, p in module.named_parameters():
            if p.requires_grad:
                print("  +", _strip(n))

    def _freeze_encoder_all(self, freeze: bool = True):
        fm = self.networks['feat_model']
        for p in fm.parameters():
            p.requires_grad = (not freeze)
        print(f"[Freeze] Encoder frozen = {freeze}")

    def _rebuild_optimizer_with_current_trainable(self):
        all_params = []
        for key, val in self.config['networks'].items():
            op = val.get('optim_params', None)
            if op is None:
                continue
            trainable = [p for p in self.networks[key].parameters() if p.requires_grad]
            if len(trainable) == 0:
                continue
            all_params.append({
                'params': trainable,
                'lr': op.get('lr', 0.001),
                'momentum': op.get('momentum', 0.9),
                'weight_decay': op.get('weight_decay', 0.0),
            })
        self.model_optimizer, self.model_optimizer_scheduler = self.init_optimizers(all_params)

    # ------------------------------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------------------------------
    def _supports_cls_only(self, module: nn.Module) -> bool:
        try:
            sig = inspect.signature(module.module.forward if isinstance(module, nn.DataParallel) else module.forward)
            return ('cls_only' in sig.parameters)
        except Exception:
            return False

    def _forward_feat(self, inputs, *, cls_only_flag: bool):
        fm = self.networks['feat_model']
        if self._supports_cls_only(fm):
            out = fm(inputs, cls_only=cls_only_flag)
        else:
            out = fm(inputs)
        if isinstance(out, (list, tuple)):
            features = out[0]
            featmaps = out[1] if len(out) > 1 else None
        else:
            features, featmaps = out, None
        return features, featmaps

    def _compute_ce_weight(self, epoch: int) -> float:
        if epoch < self.cls_start:
            return 0.0
        if self.ce_ramp <= 0:
            return float(self.ce_end_weight)
        t = min(max(epoch - self.cls_start, 0), self.ce_ramp) / float(self.ce_ramp)
        return float(self.ce_end_weight) * t

    # ------------------------------------------------------------------------------------------
    # Batch fwd/bwd/loss
    # ------------------------------------------------------------------------------------------
    def batch_forward(self, inputs, labels=None, phase='train'):
        self._last_inputs = inputs.detach().clone()

        # Stage flags
        epoch = self.current_epoch
        in_stage1 = (epoch <= self.disent_epochs)
        in_stage2 = (epoch >= self.proto_start) and (epoch < self.cls_start)
        in_stage3 = (epoch >= self.cls_start)

        # For Stage-2/3 we bypass reconstruction/style paths inside feat_model if supported
        cls_only_flag = (in_stage2 or in_stage3)

        # Forward encoder (semantic feature)
        self.features, self.feature_maps = self._forward_feat(inputs, cls_only_flag=cls_only_flag)

        # ---- Unified: compute logging/eval logits for ALL stages under no_grad() ----
        self.logits = None
        self.route_logits = None
        self.labels_cls = labels
        if 'classifier' in self.networks:
            with torch.no_grad():
                try:
                    lg, _ = self._call_classifier(self.features, labels, phase='eval', centroids_=None)
                except Exception:
                    lg = self.networks['classifier'](self.features)
                self.logits = lg

    def batch_backward(self):
        """Safe backward: only backprop when loss carries gradients."""
        if getattr(self, 'model_optimizer', None) is None:
            return
        self.model_optimizer.zero_grad()
        if getattr(self, 'criterion_optimizer', None):
            self.criterion_optimizer.zero_grad()

        do_backward = isinstance(self.loss, torch.Tensor) and self.loss.requires_grad and (self.loss.grad_fn is not None)
        if do_backward:
            self.loss.backward()
            self.model_optimizer.step()
            if getattr(self, 'criterion_optimizer', None):
                self.criterion_optimizer.step()

    def _accumulate_stage1_disent_losses(self):
        """reconstruction/KL/z-reencode/orthogonality; read caches from feat_model.module"""
        fm = self.networks['feat_model'].module if isinstance(self.networks['feat_model'], nn.DataParallel) else self.networks['feat_model']
        self.loss_rec = None
        self.loss_kl = None
        self.loss_zrec = None
        self.loss_ortho = None

        # recon L1
        x_hat = getattr(fm, 'cache_x_hat', None)
        x_inp = getattr(self, '_last_inputs', None)
        if (x_hat is not None) and (x_inp is not None) and (self.lambda_rec > 0.0):
            B = min(x_hat.size(0), x_inp.size(0))
            self.loss_rec = F.l1_loss(x_hat[:B], x_inp[:B])
            self.loss = self.loss + self.lambda_rec * self.loss_rec

        # KL
        mu = getattr(fm, 'cache_mu', None)
        lv = getattr(fm, 'cache_logvar', None)
        if (mu is not None) and (lv is not None) and (self.lambda_kl > 0.0):
            self.loss_kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            self.loss = self.loss + self.lambda_kl * self.loss_kl

        # z re-encode
        if (self.lambda_zrec > 0.0) and hasattr(fm, 'reencode_z'):
            z_re = fm.reencode_z()
            z_cur = getattr(fm, 'cache_z', None)
            if (z_re is not None) and (z_cur is not None):
                B = min(z_re.size(0), z_cur.size(0))
                self.loss_zrec = F.l1_loss(z_re[:B], z_cur[:B])
                self.loss = self.loss + self.lambda_zrec * self.loss_zrec

        # orthogonality (semantic vs style)
        if self.lambda_ortho > 0.0:
            s_vec = getattr(fm, 'cache_s_vec', None)
            z_vec = getattr(fm, 'cache_mu' if self.ortho_use_mu else 'cache_z', None)
            if (s_vec is not None) and (z_vec is not None) and s_vec.dim()==2 and z_vec.dim()==2:
                s = s_vec - s_vec.mean(dim=0, keepdim=True)
                z = z_vec - z_vec.mean(dim=0, keepdim=True)
                s = s / (s.norm(p=2, dim=1, keepdim=True) + 1e-12)
                z = z / (z.norm(p=2, dim=1, keepdim=True) + 1e-12)
                c = torch.matmul(s.t(), z) / max(1.0, float(s.size(0)))
                self.loss_ortho = (c ** 2).sum()
                self.loss = self.loss + self.lambda_ortho * self.loss_ortho

    def _call_classifier(self, features, labels, phase, centroids_=None):
        """Adapter for different classifier signatures."""
        try:
            return self.networks['classifier'](features, centroids_, phase, labels)
        except TypeError:
            return self.networks['classifier'](features, labels, None)

    def batch_loss(self, labels, epoch: int):
        """
        Stage-1: disent losses (rec/kl/zrec/ortho); classifier frozen; CMS disabled
        Stage-2: ONLY CMSLoss on L2(features); classifier frozen
        Stage-3: ONLY CE (with ramp) on classifier; encoder frozen
        """
        self.loss = torch.tensor(0.0, device=self.device)

        in_stage1 = (epoch <= self.disent_epochs)
        in_stage2 = (epoch >= self.proto_start) and (epoch < self.cls_start)
        in_stage3 = (epoch >= self.cls_start)

        # Stage-1 losses
        if in_stage1:
            self._accumulate_stage1_disent_losses()


        # Stage-2: supervised prototype contrastive (mean-shift) on projector features
        self.loss_feat = torch.tensor(0.0, device=self.device)
        if in_stage2 and ('FeatureLoss' in self.criterions) and (self.lambda_proto > 0.0):
            # Use the MLP projector from feat_model to obtain contrastive features.
            # We support both DataParallel and plain modules.
            fm = self.networks['feat_model']
            if isinstance(fm, nn.DataParallel):
                proj_feats = fm.module.projector(self.features)
            else:
                proj_feats = fm.projector(self.features)

            # proj_feats should already be L2-normalized inside projector().
            l_proto = self.criterions['FeatureLoss'](proj_feats, labels, extras=None)

            # In case the loss returns per-sample values, average them.
            if hasattr(l_proto, 'dim') and l_proto.dim() > 0:
                l_proto = l_proto.mean()

            # Apply lambda_proto and criterion weight.
            self.loss_feat = l_proto * self.lambda_proto * self.criterion_weights.get('FeatureLoss', 1.0)
            self.loss = self.loss + self.loss_feat


        # Stage-3: CE with ramp (WITH grad). Also overwrite self.logits with a detached copy for metrics.
        self.loss_perf = torch.tensor(0.0, device=self.device)
        self.ce_weight_current = self._compute_ce_weight(epoch)
        if in_stage3 and (self.ce_weight_current > 0.0) and ('PerformanceLoss' in self.criterions) and (self.lambda_ce > 0.0):
            logits, _ = self._call_classifier(self.features, labels, phase='train', centroids_=None)
            self.logits = logits.detach()  # replace logging logits so eval/train metrics are consistent in S3
            ce = self.criterions['PerformanceLoss'](logits, labels, self.features, self.networks['classifier'])
            self.loss_perf = ce * self.lambda_ce * self.criterion_weights.get('PerformanceLoss', 1.0) * self.ce_weight_current
            self.loss = self.loss + self.loss_perf

    # ------------------------------------------------------------------------------------------
    # Stage transitions (freeze policy)
    # ------------------------------------------------------------------------------------------
    def _maybe_apply_stage2_freeze(self, epoch: int):
        if self._stage2_frozen_applied:
            return
        if (epoch >= self.proto_start) and (epoch < self.cls_start) and self.freeze_low_at_proto:
            print(f"[Stage Switch] Entering Stage-2 at epoch {epoch}: freeze low (layer1-3), train layer4+ and head.")
            fm = self.networks['feat_model']
            self._apply_regex_freeze(
                fm,
                self.proto_freeze_regex,
                self.proto_unfreeze_regex
            )
            # classifier still frozen
            self._set_classifier_trainable(False)
            self._rebuild_optimizer_with_current_trainable()
            self._stage2_frozen_applied = True

    def _maybe_apply_stage3_freeze(self, epoch: int):
        if self._stage3_frozen_applied:
            return
        if epoch >= self.cls_start:
            print(f"[Stage Switch] Entering Stage-3 at epoch {epoch}: freeze entire encoder; enable classifier.")
            self._set_classifier_trainable(True)
            self._freeze_encoder_all(True)
            self._rebuild_optimizer_with_current_trainable()
            self._stage3_frozen_applied = True

    # ------------------------------------------------------------------------------------------
    # Public train
    # ------------------------------------------------------------------------------------------
    def train(self):
        print_write(['Phase: train'], self.log_file)
        time.sleep(0.2)
        print_write(['Do shuffle??? --- ', self.do_shuffle], self.log_file)

        best_model_weights = {
            'feat_model': copy.deepcopy(self.networks['feat_model'].state_dict()),
            'classifier': copy.deepcopy(self.networks.get('classifier', nn.Identity()).state_dict()
                                        if 'classifier' in self.networks else {})
        }
        best_acc, best_epoch, best_centroids = 0.0, 0, None
        end_epoch = self.training_opt['num_epochs']
        first_batch = True

        # Stage-1 init: encoder trainable, classifier frozen
        self._set_classifier_trainable(False)
        self._freeze_encoder_all(False)
        self._rebuild_optimizer_with_current_trainable()

        for epoch in range(1, end_epoch + 1):
            self.current_epoch = epoch
            self._maybe_apply_stage2_freeze(epoch)
            self._maybe_apply_stage3_freeze(epoch)

            # classifier mode per stage
            clf = self.networks.get('classifier', None)
            if clf is not None:
                if epoch < self.cls_start:
                    clf.eval()
                else:
                    clf.train()

            for m in self.networks.values():
                m.train()
            torch.cuda.empty_cache()

            total_preds, total_labels = [], []
            sum_rec = sum_kl = sum_zrec = sum_ortho = 0.0
            sum_proto = sum_ce = sum_total = 0.0
            n_loss_steps = 0
            step = 0

            for inputs, labels, indexes in self.data['train']:
                step += 1
                if step == self.epoch_steps:
                    break
                if self.do_shuffle:
                    inputs, labels = self.shuffle_batch(inputs, labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if first_batch:
                    print_write([f"Batch size is {inputs.size(0)}"], self.log_file)
                    first_batch = False

                with torch.set_grad_enabled(True):
                    self.batch_forward(inputs, labels, phase='train')
                    self.batch_loss(labels, epoch)
                    self.batch_backward()

                    rec_i   = (self.loss_rec.item()   if getattr(self, 'loss_rec',   None) is not None else 0.0)
                    kl_i    = (self.loss_kl.item()    if getattr(self, 'loss_kl',    None) is not None else 0.0)
                    zrec_i  = (self.loss_zrec.item()  if getattr(self, 'loss_zrec',  None) is not None else 0.0)
                    ortho_i = (self.loss_ortho.item() if getattr(self, 'loss_ortho', None) is not None else 0.0)
                    proto_i = (self.loss_feat.item()  if hasattr(self, 'loss_feat')  else 0.0)
                    ce_i    = (self.loss_perf.item()  if hasattr(self, 'loss_perf')  else 0.0)
                    tot_i   = float(self.loss.item()) if isinstance(self.loss, torch.Tensor) else float(self.loss)

                    sum_rec += rec_i; sum_kl += kl_i; sum_zrec += zrec_i; sum_ortho += ortho_i
                    sum_proto += proto_i; sum_ce += ce_i; sum_total += tot_i; n_loss_steps += 1

                    if self.logits is not None:
                        _, preds = torch.max(self.logits, 1)
                        total_preds.append(torch2numpy(preds))
                        total_labels.append(torch2numpy(self.labels_cls))

                    if step % self.training_opt['display_step'] == 0:
                        minibatch_acc = 0.0
                        if self.logits is not None:
                            _, preds_dbg = torch.max(self.logits, 1)
                            minibatch_acc = mic_acc_cal(preds_dbg, self.labels_cls)

                        stage_str = ("S1" if epoch <= self.disent_epochs else
                                     ("S3" if epoch >= self.cls_start else "S2"))
                        print_str = [
                            f'Stage: {stage_str}',
                            'Epoch: [%d/%d]' % (epoch, self.training_opt['num_epochs']),
                            'Step: %5d' % (step),
                            f'CE_w: {self.ce_weight_current:.3f}',
                            'Minibatch_loss_CMS: %.3f' % (proto_i),
                            'Minibatch_loss_CE: %.3f' % (ce_i),
                            'Rec: %.3f KL: %.3f Zrec: %.3f Ortho: %.3f' % (rec_i, kl_i, zrec_i, ortho_i),
                            'Minibatch_accuracy_micro: %.3f' % (minibatch_acc)
                        ]
                        print_write(print_str, self.log_file)

                        # log scalar losses
                        self.logger.log_loss({
                            'Epoch': epoch, 'Step': step, 'Total': tot_i,
                            'CE': ce_i, 'CMS': proto_i,
                            'rec': rec_i, 'kl': kl_i, 'zrec': zrec_i, 'ortho': ortho_i,
                            'CE_w': self.ce_weight_current
                        })

                        # Stage-2 batch class histogram (diagnostics for tail)
                        if (epoch >= self.proto_start) and (epoch < self.cls_start):
                            with torch.no_grad():
                                binc = torch.bincount(labels, minlength=int(self.training_opt['num_classes'])).cpu().numpy().tolist()
                            print_write([f'[S2] batch class hist: {binc}'], self.log_file)

                # optional sampler priority
                if hasattr(self.data['train'].sampler, 'update_weights') and (self.logits is not None):
                    ptype = getattr(self.data['train'].sampler, 'ptype', 'score')
                    ws = get_priority(ptype, self.logits.detach(), self.labels_cls)
                    inlist = [indexes.cpu().numpy(), ws]
                    if self.training_opt.get('sampler', {}).get('type') == 'ClassPrioritySampler':
                        inlist.append(self.labels_cls.cpu().numpy())
                    self.data['train'].sampler.update_weights(*inlist)

            # sched step
            self.model_optimizer_scheduler.step()
            if getattr(self, 'criterion_optimizer_scheduler', None):
                self.criterion_optimizer_scheduler.step()

            if hasattr(self.data['train'].sampler, 'get_weights'):
                self.logger.log_ws(epoch, self.data['train'].sampler.get_weights())
            if hasattr(self.data['train'].sampler, 'reset_weights'):
                self.data['train'].sampler.reset_weights(epoch)

            if n_loss_steps > 0:
                avg_rec = sum_rec / n_loss_steps; avg_kl = sum_kl / n_loss_steps
                avg_zrec = sum_zrec / n_loss_steps; avg_ortho = sum_ortho / n_loss_steps
                avg_proto = sum_proto / n_loss_steps; avg_ce = sum_ce / n_loss_steps
                avg_total = sum_total / n_loss_steps

                epoch_summary = [f"\n[Epoch {epoch:03d} summary] "
                                 f"Total: {avg_total:.4f} | CE: {avg_ce:.4f} | CMS: {avg_proto:.4f} | "
                                 f"Rec: {avg_rec:.4f} | KL: {avg_kl:.4f} | Zrec: {avg_zrec:.4f} | Ortho: {avg_ortho:.4f}\n"]
                print_write(epoch_summary, self.log_file)
                self.logger.log_loss({
                    'Epoch': epoch, 'Step': -1,
                    'Total': avg_total, 'CE': avg_ce, 'CMS': avg_proto,
                    'rec': avg_rec, 'kl': avg_kl, 'zrec': avg_zrec, 'ortho': avg_ortho,
                    'CE_w': self.ce_weight_current
                })

            # training confusion (when logits exist)
            rsls_train = {'train_all': 0., 'train_many': 0., 'train_median': 0., 'train_low': 0.}
            if len(total_preds) > 0:
                rsls_train = self.eval_with_preds(total_preds, total_labels)
                try:
                    y_pred = np.concatenate(total_preds)
                    y_true = np.concatenate(total_labels)
                    cm = self._confusion_matrix(y_true.astype(int), y_pred.astype(int),
                                                num_classes=int(self.training_opt['num_classes']))
                    out_prefix = os.path.join(self.training_opt['log_dir'], f'train_epoch{epoch:03d}')
                    class_names = [str(i) for i in range(int(self.training_opt['num_classes']))]
                    self._save_confusion_csv_png(cm, out_prefix, class_names)
                    self._print_confusion(cm, class_names, title=f"[Train] Confusion Matrix (epoch {epoch})")
                except Exception as e:
                    print(f"[WARN] train confusion failed: {e}")

            rsls_eval = self.eval(phase='val')
            self.logger.log_acc({**{'epoch': epoch}, **rsls_train, **rsls_eval})

            if getattr(self, 'eval_acc_mic_top1', 0.0) > best_acc:
                best_epoch = epoch
                best_acc = float(self.eval_acc_mic_top1)
                best_centroids = self.centroids
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                if 'classifier' in self.networks:
                    best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

            print('===> Saving checkpoint')
            self.save_latest(epoch)

        print('\nTraining Complete.')
        print_write([f'Best validation accuracy is {best_acc:.3f} at epoch {best_epoch}'], self.log_file)
        self.save_model(end_epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)
        self.reset_model(best_model_weights)
        self.eval('test' if 'test' in self.data else 'val')
        print('Done')

    # ------------------------------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------------------------------
    def shuffle_batch(self, x, y):
        idx = torch.randperm(x.size(0))
        return x[idx], y[idx]

    def eval_with_preds(self, preds, labels):
        n_total = sum([len(p) for p in preds])
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)
        rsl = {'train_all': 0., 'train_many': 0., 'train_median': 0., 'train_low': 0.}
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            n_top1_many, n_top1_median, n_top1_low = shot_acc(normal_preds, normal_labels, self.data['train'])
            rsl['train_all'] += len(normal_preds) / n_total * n_top1
            rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
            rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
            rsl['train_low'] += len(normal_preds) / n_total * n_top1_low
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = list(map(np.concatenate, [mixup_preds * 2, mixup_labels1 + mixup_labels2, mixup_ws]))
            mixup_ws = np.concatenate([mixup_ws, 1 - mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, n_top1_median, n_top1_low = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, self.data['train'])
            rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
            rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low
        print_write([
            '\n Training acc Top1: %.3f \n' % (rsl['train_all']),
            'Many_top1: %.3f' % (rsl['train_many']),
            'Median_top1: %.3f' % (rsl['train_median']),
            'Low_top1: %.3f' % (rsl['train_low']),
            '\n'
        ], self.log_file)
        return rsl

    @staticmethod
    def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
        return cm

    @staticmethod
    def _precision_recall_f1_per_class(cm: np.ndarray):
        eps = 1e-12
        tp = np.diag(cm).astype(np.float64)
        pred_pos = cm.sum(axis=0).astype(np.float64)
        true_pos = cm.sum(axis=1).astype(np.float64)
        prec = tp / np.maximum(pred_pos, eps)
        rec = tp / np.maximum(true_pos, eps)
        f1 = 2 * (prec * rec) / np.maximum(prec + rec, eps)
        return prec, rec, f1

    @staticmethod
    def _save_confusion_csv_png(cm: np.ndarray, out_prefix: str, class_names: List[str]):
        import csv
        csv_path = out_prefix + "_confmat.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + class_names)
            for i, row in enumerate(cm):
                w.writerow([class_names[i]] + list(map(int, row)))
        if _HAS_MPL:
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(cm, interpolation='nearest')
            ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha='center', va='center')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout(); fig.savefig(out_prefix + "_confmat.png", dpi=200); plt.close(fig)

    def _print_confusion(self, cm: np.ndarray, class_names: List[str], title: str):
        width = max(6, max(len(n) for n in class_names) + 2)
        head = "true\\pred" + "".join(f"{name:>{width}}" for name in class_names)
        print_write(["\n" + title], self.log_file); print_write([head], self.log_file)
        for i, row in enumerate(cm.astype(int)):
            line = f"{class_names[i]:>9}" + "".join(f"{v:>{width}d}" for v in row)
            print_write([line], self.log_file)
        print_write([""], self.log_file)

    def _to_numpy1d(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, (list, tuple)):
            try:
                return np.array(x)
            except Exception:
                return np.array(list(x))
        return np.array(x)

    # ------------------------------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------------------------------
    def eval(self, phase='val', openset=False, save_feat=False):
        print_write([f'Phase: {phase}'], self.log_file)
        time.sleep(0.05)

        if openset:
            print('Under openset test mode. Open threshold is %.1f' % self.training_opt['open_threshold'])

        torch.cuda.empty_cache()
        for m in self.networks.values():
            m.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_paths = np.empty(0, dtype=object)

        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        get_feat_only = save_feat

        for inputs, labels, third in tqdm(self.data[phase]):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            third_np = self._to_numpy1d(third)

            with torch.set_grad_enabled(False):
                self.batch_forward(inputs, labels, phase=phase)

                if not get_feat_only:
                    if self.logits is not None:
                        self.total_logits = torch.cat((self.total_logits, self.logits))
                        self.total_labels = torch.cat((self.total_labels, labels))
                        self.total_paths = np.concatenate((self.total_paths, third_np))
                else:
                    logits_all.append(None if self.logits is None else self.logits.cpu().numpy())
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(third_np)

        if get_feat_only:
            typ = 'feat'
            name = {'train_plain': 'train', 'val': 'val', 'test': 'test'}[phase] + f'{typ}_all.pkl'
            fname = os.path.join(self.training_opt['log_dir'], name)
            print('===> Saving feats to ' + fname)
            with open(fname, 'wb') as f:
                pickle.dump({
                    'feats': np.concatenate(feats_all),
                    'labels': np.concatenate(labels_all),
                    'idxs': np.concatenate(idxs_all),
                }, f, protocol=4)
            return

        if self.total_logits.shape[0] == 0:
            self.eval_acc_mic_top1 = 0.0
            self.many_acc_top1 = 0.0
            self.median_acc_top1 = 0.0
            self.low_acc_top1 = 0.0
            print_write(['\n[Eval] No logits in this phase; skip classification metrics.\n'], self.log_file)
            return {
                phase + '_all': 0.0, phase + '_macro_acc': 0.0, phase + '_macro_f1': 0.0,
                phase + '_weighted_f1': 0.0, phase + '_many': 0.0, phase + '_median': 0.0, phase + '_low': 0.0,
            }

        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)
        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1], self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        mask = (self.total_labels != -1)
        y_true = self.total_labels[mask]
        y_pred = preds[mask]
        num_classes = int(self.training_opt['num_classes'])
        class_names = [str(i) for i in range(num_classes)]
        self.eval_acc_mic_top1 = mic_acc_cal(y_pred, y_true)

        y_true_np = y_true.detach().cpu().numpy().astype(int)
        y_pred_np = y_pred.detach().cpu().numpy().astype(int)
        cm = self._confusion_matrix(y_true_np, y_pred_np, num_classes=num_classes)

        prec, rec, f1 = self._precision_recall_f1_per_class(cm)
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        macro_acc = float(np.nanmean(acc_per_class))
        macro_f1 = float(np.nanmean(f1))
        weights = cm.sum(axis=1).astype(np.float64)
        weights = weights / np.maximum(weights.sum(), 1e-12)
        weighted_f1 = float(np.sum(f1 * weights))

        out_prefix = os.path.join(self.training_opt['log_dir'], phase)
        try:
            self._save_confusion_csv_png(cm, out_prefix, class_names)
            self._print_confusion(cm, class_names, title=f"[{phase}] Confusion Matrix")
        except Exception as e:
            print(f"[WARN] save confusion failed: {e}")

        print_write(['\n\n', f'Phase: {phase}', '\n\n',
                     'Accuracy (micro top1): %.3f' % (self.eval_acc_mic_top1), '\n',
                     'Macro accuracy: %.3f' % (macro_acc), '\n',
                     'Macro F1: %.3f | Weighted F1: %.3f' % (macro_f1, weighted_f1), '\n'
                     ], self.log_file)

        self.many_acc_top1, self.median_acc_top1, self.low_acc_top1, self.cls_accs = shot_acc(
            y_pred, y_true, self.data['train'], acc_per_cls=True
        )

        rsl = {
            phase + '_all': float(self.eval_acc_mic_top1),
            phase + '_macro_acc': float(macro_acc),
            phase + '_macro_f1': float(macro_f1),
            phase + '_weighted_f1': float(weighted_f1),
            phase + '_many': float(self.many_acc_top1),
            phase + '_median': float(self.median_acc_top1),
            phase + '_low': float(self.low_acc_top1),
        }

        if phase == 'test':
            with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
                pickle.dump(self.cls_accs, f)
        return rsl

    # ------------------------------------------------------------------------------------------
    # Centroids / IO
    # ------------------------------------------------------------------------------------------
    def centroids_cal(self, data, save_all=False):
        centroids = torch.zeros(self.training_opt['num_classes'],
                                self.training_opt['feature_dim']).to(self.device)
        print('Calculating centroids.')
        torch.cuda.empty_cache()
        for m in self.networks.values():
            m.eval()

        feats_all, labels_all, idxs_all = [], [], []
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                out = self.networks['feat_model'](inputs)
                features = out[0] if isinstance(out, (list, tuple)) else out
                for i in range(len(labels)):
                    centroids[labels[i]] += features[i]
                if save_all:
                    feats_all.append(features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(self._to_numpy1d(idxs))

        cc = torch.tensor(class_count(data)).float().unsqueeze(1).to(self.device)
        centroids = centroids / torch.clamp(cc, min=1.0)
        return centroids

    def reset_model(self, model_state):
        for key, m in self.networks.items():
            if key not in ('classifier', 'feat_model'):
                continue
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in m.state_dict()}
            m.load_state_dict(weights)

    def load_model(self, model_dir=None):
        model_dir = self.training_opt['log_dir'] if model_dir is None else model_dir
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')

        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))
        checkpoint = torch.load(model_dir, map_location='cpu')

        if 'state_dict_best' in checkpoint:
            model_state = checkpoint['state_dict_best']
            self.centroids = checkpoint.get('centroids', None)
            for key, m in self.networks.items():
                if (not self.test_mode) and ('DotProductClassifier' in self.config['networks'][key]['def_file']):
                    print('Skipping classifier initialization')
                    continue
                weights = model_state.get(key, {})
                weights = {k: weights[k] for k in weights if k in m.state_dict()}
                x = m.state_dict(); x.update(weights); m.load_state_dict(x)
        else:
            model_state = checkpoint
            self.centroids = None
            for key, m in self.networks.items():
                if (not self.test_mode) and ('DotProductClassifier' in self.config['networks'][key]['def_file']):
                    print('Skipping classifier initialization')
                    continue
                weights = {}
                for k in model_state:
                    weights['module.' + k] = model_state[k] if 'module.' not in k else model_state[k]
                weights = {k: weights[k] for k in weights if k in m.state_dict()}
                x = m.state_dict(); x.update(weights); m.load_state_dict(x)

    def save_latest(self, epoch):
        model_weights = {
            'feat_model': copy.deepcopy(self.networks['feat_model'].state_dict()),
            'classifier': copy.deepcopy(self.networks.get('classifier', nn.Identity()).state_dict()
                                        if 'classifier' in self.networks else {})
        }
        model_states = {'epoch': epoch, 'state_dict': model_weights}
        model_dir = os.path.join(self.training_opt['log_dir'], 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)

    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):
        model_states = {
            'epoch': epoch, 'best_epoch': best_epoch,
            'state_dict_best': best_model_weights, 'best_acc': float(best_acc), 'centroids': centroids
        }
        model_dir = os.path.join(self.training_opt['log_dir'], 'final_model_checkpoint.pth')
        torch.save(model_states, model_dir)
