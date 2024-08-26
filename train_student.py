import argparse
import time
import datetime
import os
import shutil
import sys

import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from losses.loss import SegCrossEntropyLoss
from models.model_zoo import get_segmentation_model

from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from dataset.datasets import CSTrainValSet
from utils.flops import cal_multi_adds, cal_param_size


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')

    # model and dataset
    parser.add_argument('--student-model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--student-backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='potsdam',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='vaihingen256_stride/',
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 512], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,  # 8
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='ignore label')

    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')

    parser.add_argument('--pixel-memory-size', type=int, default=20000)
    parser.add_argument('--region-memory-size', type=int, default=2000)
    parser.add_argument('--region-contrast-size', type=int, default=1024)
    parser.add_argument('--pixel-contrast-size', type=int, default=4096)

    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    parser.add_argument("--contrast-kd-temperature", type=float, default=1.0,
                        help="similarity distribution KD temperature")
    parser.add_argument("--contrast-temperature", type=float, default=0.1, help="similarity distribution temperature")

    parser.add_argument("--lambda-kd", type=float, default=1., help="lambda_kd")
    parser.add_argument("--lambda-minibatch-pixel", type=float, default=1., help="lambda mini-batch-based pixel")
    parser.add_argument("--lambda-memory-pixel", type=float, default=0.1, help="lambda memory-based pixel")
    parser.add_argument("--lambda-memory-region", type=float, default=0.1, help="lambda memory-based region")

    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
    parser.add_argument('--student-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')

    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # ..dcs
    path_now = os.path.dirname(os.path.abspath(__file__))

    args.data = os.path.join(path_now, args.data)
    # dcs~/.torch/models  ../runs/logs/  none dcs/
    args.save_dir = os.path.join(path_now, args.save_dir)
    args.log_dir = os.path.join(path_now, args.log_dir)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # not
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.student_backbone.startswith('resnet'):
        args.aux = True  # resnet startswith() 方法用于检查字符串是否是以指定子字符串开头
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        train_dataset = CSTrainValSet(args.data,
                                      list_path='vaihingen256_stride/train.lst',
                                      max_iters=args.max_iterations * args.batch_size,
                                      crop_size=args.crop_size, scale=True, mirror=True)
        val_dataset = CSTrainValSet(args.data,
                                    list_path='vaihingen256_stride/test.lst',
                                    crop_size=(512, 512), scale=False, mirror=False)

        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)

        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d


        self.s_model = get_segmentation_model(model=args.student_model,
                                              backbone=args.student_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base=args.student_pretrained_base,
                                              pretrained='None',
                                              aux=True,
                                              norm_layer=nn.BatchNorm2d,
                                              num_class=train_dataset.num_class).to(self.device)


        self.s_model.eval()

        # resume checkpoint if needed

        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.s_model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion

        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        params_list = nn.ModuleList([])
        params_list.append(self.s_model)

        self.optimizer = torch.optim.SGD(params_list.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model,
                                                               device_ids=[args.local_rank],
                                                               output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0

    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr * ((1 - float(iter) / max_iter) ** (power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters  # 1600
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(args.max_iterations))
        scaler = torch.cuda.amp.GradScaler()

        self.s_model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            images = images.to(self.device)
            targets = targets.long().to(self.device)

            self.optimizer.zero_grad()  # 清空过往梯度
            with torch.cuda.amp.autocast():
                s_outputs = self.s_model(images)
                pred_np_discrete = torch.argmax(s_outputs[0], dim=1).data.cpu().numpy()

                if self.args.aux:
                    #
                    task_loss = self.criterion(s_outputs[0], targets) + 0.4 * self.criterion(s_outputs[1], targets)
                else:
                    task_loss = self.criterion(s_outputs[0], targets)

                losses = task_loss

            lr = self.adjust_lr(base_lr=args.lr, iter=iteration - 1, max_iter=args.max_iterations, power=0.9)
            scaler.scale(losses).backward()  # 反向传播，计算当前梯度
            scaler.step(self.optimizer)  # 根据梯度更新网络参数
            scaler.update()  # scaler factor更新

            task_losses_reduced = task_loss

            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} " \
                    "|| Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'],
                        task_losses_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

                writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], iteration)
                writer.add_scalar('Train/Loss_task', task_losses_reduced.item(), iteration)

            if iteration % save_per_iters == 0 and save_to_disk:  # 1600
                save_checkpoint(self.s_model, self.args, is_best=False, iteration= iteration)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(iteration)
                self.s_model.train()

        save_checkpoint(self.s_model, self.args, is_best=False, iteration= iteration)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / args.max_iterations))

    def validation(self, iteration):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(outputs[0], (H, W), mode='bilinear', align_corners=True)

            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).cuda().to(args.local_rank)
            sum_total_label = torch.tensor(self.metric.total_label).cuda().to(args.local_rank)
            sum_total_inter = torch.tensor(self.metric.total_inter).cuda().to(args.local_rank)
            sum_total_union = torch.tensor(self.metric.total_union).cuda().to(args.local_rank)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label)
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            #mIoU = IoU.mean().item()
           #mIoU = np.nanmean(IoU[:-1])
            mIoU = np.nanmean(IoU[:-1].cpu().numpy())
            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                pixAcc.item() * 100, mIoU * 100))

            writer.add_scalar('Test/pixAcc', pixAcc.item() * 100, iteration)
            writer.add_scalar('Test/mIoU', mIoU * 100, iteration)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        writer.add_scalar('Test/mIoU_Best', self.best_pred * 100, iteration)

        if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
            save_checkpoint(self.s_model, self.args, is_best, iteration=iteration)
        synchronize()


def save_checkpoint(model, args, is_best=False, iteration=0):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'kd_{}_{}_{}_{}.pth'.format(args.student_model, args.student_backbone, args.dataset, iteration)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'kd_{}_{}_{}_best_model.pth'.format(args.student_model, args.student_backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1

    if (args.distributed and args.local_rank == 0):
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
    writer = SummaryWriter(args.save_dir)

    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)  # default 0
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        ## 用dist.barrier()来同步不同进程间的快慢
        synchronize()
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.student_model,  args.student_backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
