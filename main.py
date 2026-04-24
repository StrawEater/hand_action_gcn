#!/usr/bin/env python
from __future__ import print_function
import argparse
import json
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.autograd import Variable
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import random
import copy
import inspect
import torch.backends.cudnn as cudnn


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_joint_mask(x, mask_ratio, depth):
    """Randomly creates a mask of non overlapping segments over verterces and frames equal in both hands.

    Returns:
        joint_mask: (N, 1, T, V, 1) float tensor, 0=not choosen 1=choosen
    """
    N, C, T, V, M = x.shape

    one_hand = V//2
    total_frames_segments = T // depth
    total_keypoints_segments = total_frames_segments * one_hand

    n_mask = max(1, int(mask_ratio * total_keypoints_segments))
    joint_mask = torch.zeros(N, 1, T, one_hand, 1, device=x.device)
    
    for i in range(N):
        idx = torch.randperm(total_keypoints_segments, device=x.device)[:n_mask]

        time_frame_segment_idx = idx // one_hand
        joint_segment_idx = idx % one_hand

        for j in range(n_mask):
            start = int(time_frame_segment_idx[j]) * depth
            end = start + depth
            joint_idx = int(joint_segment_idx[j])
            joint_mask[i, 0, start:end, joint_idx, 0] = 1
    
    both_hand_mask = joint_mask.repeat_interleave(2, dim=3)  # dim=3 is V

    return both_hand_mask

def apply_joint_zero_mask(x, mask_ratio, depth):
    
    both_hand_mask = create_joint_mask(x, mask_ratio, depth)
    hide_mask = 1 - both_hand_mask 

    return x * hide_mask, both_hand_mask

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Shift Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--warm_up_epoch', default=0)
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='label smoothing factor for cross-entropy loss')
    parser.add_argument('--mask_ratio', type=float, default=0.0,
                        help='fraction of joints to mask per sample for reconstruction loss (0 = disabled)')
    parser.add_argument('--mask_depth', type=int, default=3,
                        help='number of frames per masked segment')
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='weight of reconstruction loss relative to classification loss')
    parser.add_argument('--t_max_mult', type=float, default=1.0,
                        help='multiplier for CosineAnnealingLR T_max')
    parser.add_argument('--freeze_temporal', type=str2bool, default=False,
                        help='freeze all temporal layers (Shift_tcn and stride residuals)')
    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def import_class(name):
    import importlib
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def load_yaml(parser, yaml_config):
    
    parser_args = parser.parse_args()

    with open(yaml_config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.SafeLoader)
        
        key = vars(parser_args).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        
        parser.set_defaults(**default_arg)
    
    return parser.parse_args()


def make_arg_from_dict(config_dict):
    """Build an arg namespace from a config dict (used by sweep.py)."""
    parser = get_parser()
    arg = parser.parse_args([])
    for k, v in config_dict.items():
        setattr(arg, k, v)
    return arg


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg, non_interactive=False):

        arg.model_saved_name = "./save_models/"+arg.Experiment_name
        arg.work_dir = "./work_dir/"+arg.Experiment_name
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args.get('debug', False):
                if os.path.isdir(arg.model_saved_name):
                    if non_interactive:
                        print('log_dir: ', arg.model_saved_name, 'already exist — skipping prompt (non-interactive mode)')
                    else:
                        print('log_dir: ', arg.model_saved_name, 'already exist')
                        answer = input('delete it? y/n:')
                        if answer == 'y':
                            shutil.rmtree(arg.model_saved_name)
                            print('Dir removed: ', arg.model_saved_name)
                            input('Refresh the website of tensorboard by pressing any keys')
                        else:
                            print('Dir not removed: ', arg.model_saved_name)

        self.global_step = 0
        self.load_model()
        self.load_data()
        self.load_optimizer()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def create_weight_sample(self):
        class_sample_count = np.array([len(np.where(self.train_labels == t)[0]) for t in np.unique(self.train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.train_labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        return WeightedRandomSampler(samples_weight, len(samples_weight))


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            train_dataset = Feeder(**self.arg.train_feeder_args)
            
            self.train_labels = train_dataset.label
            weight_sampler = self.create_weight_sample()

            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed,
                sampler=weight_sampler)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        
        self.model = Model(**self.arg.model_args).cuda(output_device)

        if self.arg.phase == 'train':
                
            label_smoothing = getattr(self.arg, 'label_smoothing', 0.1)
            self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing).cuda(self.output_device)
            
        else:
            self.loss = nn.CrossEntropyLoss().cuda(output_device)

        print(self.arg.weights)

        if self.arg.weights:

            device = next(self.model.parameters()).device
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            
            checkpoint = torch.load(self.arg.weights, map_location=device, weights_only=True)
            checkpoint = OrderedDict([(k.split('module.')[-1], v) for k, v in checkpoint.items()])
            
            model_state = self.model.state_dict()
            skipped = []
            filtered = OrderedDict()
            
            for k, v in checkpoint.items():
                if k not in model_state:
                    skipped.append(f'{k} (not in model)')
                elif v.shape != model_state[k].shape:
                    skipped.append(f'{k}: checkpoint {list(v.shape)} vs model {list(model_state[k].shape)}')
                else:
                    filtered[k] = v
            
            model_state.update(filtered)
            result = self.model.load_state_dict(model_state, strict=False)
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)
            print("Skipped:", skipped)

            # if '.pkl' in self.arg.weights:
            #     with open(self.arg.weights, 'r') as f:
            #         weights = pickle.load(f)
            # else:
            #     weights = torch.load(self.arg.weights, weights_only=True)

            # weights = OrderedDict(
            #     [[k.split('module.')[-1],
            #       v.cuda(output_device)] for k, v in weights.items()])

            # print("output_device:", output_device)
            # print("model device:", next(self.model.parameters()).device)

            # weights_raw = torch.load(self.arg.weights, weights[_only=True)
            # first_key = next(iter(weights_raw.keys()))
            # first_tensor = weights_raw[first_key]
            # model_tensor = self.model.state_dict()[first_key.split('module.')[-1]]

            # print(f"Key: {first_key}")
            # print(f"Checkpoint values: {first_tensor.flatten()[:5]}")
            # print(f"Model values:      {model_tensor.flatten()[:5]}")
            # print(f"Are identical:     {torch.equal(first_tensor.cpu(), model_tensor.cpu())}")


            # state = copy.deepcopy(self.model.state_dict())

            # skipped = []
            # for k, v in weights.items():
            #     if k not in state:
            #         skipped.append(f'{k} (not in model)')
            #     elif v.shape != state[k].shape:
            #         skipped.append(f'{k}: pretrained {list(v.shape)} vs model {list(state[k].shape)}')
            #     else:
            #         state[k] = v

            # before = next(self.model.parameters()).clone()
            # result = self.model.load_state_dict(state, strict=False)
            # print("Missing keys:", result.missing_keys)
            # print("Unexpected keys:", result.unexpected_keys)
            # after = next(self.model.parameters())
            # print("Weights changed:", not torch.equal(before, after))

            # result = self.model.load_state_dict(state, strict=False)
            # print("Missing keys:", result.missing_keys)
            # print("Unexpected keys:", result.unexpected_keys)

        if getattr(self.arg, 'freeze_temporal', False):
            self._freeze_temporal_layers()

    def _freeze_temporal_layers(self):
        """Freeze all Shift_tcn and stride-residual (tcn) parameters."""
        frozen = []
        for name, param in self.model.named_parameters():
            if '.tcn1.' in name or '.residual.' in name:
                param.requires_grad = False
                frozen.append(name)
        self.print_log(f'Froze {len(frozen)} temporal layer parameters (tcn1 + residuals).')

    def load_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        
        warmup_epochs = self.arg.warm_up_epoch
        steps_per_epoch = len(self.data_loader['train'])
        total_epochs = self.arg.num_epoch
        warmup_steps = steps_per_epoch * warmup_epochs
        cosine_steps = steps_per_epoch * (total_epochs - warmup_epochs)
        
        warmup_sched = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=int(cosine_steps * self.arg.t_max_mult), eta_min=1e-6)
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            os.makedirs(self.arg.work_dir+'/eval_results')
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)


    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        

        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        mask_ratio = getattr(self.arg, 'mask_ratio', 0.0)
        mask_depth = getattr(self.arg, 'mask_depth', 3.0)
        recon_weight = getattr(self.arg, 'recon_weight', 1.0)

        
        # if epoch > self.arg.warm_up_epoch:
        #     dist = epoch - self.arg.warm_up_epoch
            
        #     last_epoch_dist = 35
        #     coef = (last_epoch_dist - dist) / last_epoch_dist
            
        #     label_smoothing = 0.25 * (coef)
            
        #     self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing).cuda(self.output_device)



        with torch.set_grad_enabled(True):

            for batch_idx, (data, label, index) in enumerate(process):
                self.global_step += 1
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                timer['dataloader'] += self.split_time()

                # forward
                start = time.time()
                
                if mask_ratio > 0 and epoch <= self.arg.warm_up_epoch:
                    original_data = data.clone()
                    data_masked, joint_mask = apply_joint_zero_mask(data, mask_ratio, mask_depth)

                    output = self.model(data, return_recon=False)
                    _, recon = self.model(data_masked, return_recon=True)

                    if recon is None:
                        raise RuntimeError(
                            f"mask_ratio > 0 but {type(self.model).__name__} returned "
                            "recon=None. Ensure the model implements reconstruction output."
                        )

                    cls_loss = self.loss(output, label)

                    diff_abs = torch.abs(recon - original_data)
                    n_masked = joint_mask.sum()

                    recon_loss = (diff_abs * joint_mask).sum() / n_masked

                    loss = cls_loss + recon_weight * recon_loss

                else:

                    output = self.model(data, return_recon=False)
                    cls_loss = self.loss(output, label)
                    recon_loss = torch.tensor(0.0)
                    loss = cls_loss

                network_time = time.time() - start

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
                
                loss_value.append(loss.data)
                timer['model'] += self.split_time()

                _, predict_label = torch.max(output.data, 1)
                acc = torch.mean((predict_label == label.data).float())

                self.lr = self.optimizer.param_groups[0]['lr']

                if self.global_step % self.arg.log_interval == 0:
                    self.print_log(
                        '\tBatch({}/{}) done. Loss: {:.4f} (cls:{:.4f} recon:{:.4f})  lr:{:.6f}  t:{:.3f}s'.format(
                            batch_idx, len(loader), loss.data, cls_loss.data,
                            recon_loss.data if mask_ratio > 0 else 0.0, self.lr, network_time))
        
        timer['statistics'] += self.split_time()


    def eval(self, epoch, save_score=False, loader_name='test'):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))

        loss_value = []
        score_frag = []
        process = tqdm(self.data_loader[loader_name])

        for batch_idx, (data, label, index) in enumerate(process):
            data = data.float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)

            with torch.no_grad():
                output = self.model(data)
                loss = self.loss(output, label)

            score_frag.append(output.cpu().numpy())
            loss_value.append(loss.item())

        score = np.concatenate(score_frag)
        accuracy = self.data_loader[loader_name].dataset.top_k(score, 1)

        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.print_log('New best accuracy: {:.2f}%'.format(accuracy * 100))

        # Structured per-epoch metrics for sweep analysis
        metrics_path = os.path.join(self.arg.work_dir, 'metrics.jsonl')
        with open(metrics_path, 'a') as f:
            json.dump({'epoch': epoch + 1, 'top1': float(accuracy * 100)}, f)
            f.write('\n')

        self.print_log('Eval Accuracy: {:.2f}% | Model: {}'.format(
            accuracy * 100, self.arg.model_saved_name))
        self.print_log('Mean {} loss of {} batches: {:.4f}'.format(
            loader_name, len(self.data_loader[loader_name]), np.mean(loss_value)))
        for k in self.arg.show_topk:
            self.print_log('Top{}: {:.2f}%'.format(
                k, 100 * self.data_loader[loader_name].dataset.top_k(score, k)))








    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                self.train(epoch, save_model=True)

                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name='test')

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name='test')
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    arg = parser.parse_args()
    
    if arg.config is not None:
        arg = load_yaml(parser, arg.config)

    init_seed(0)
    processor = Processor(arg)
    processor.start()
