from __future__ import print_function

import os
import sys
import argparse

import numpy as np
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"]="0"
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_zoo import get_segmentation_model
from utils.score_prediction import SegmentationMetric
from utils.visualize import get_color_pallete
from utils.logger import setup_logger
from utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from dataset.cityscapes import CSValSet
from dataset.camvid import CamvidValSet
from dataset.voc import VOCDataValSet
from utils.flops import cal_multi_adds, cal_param_size


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation validation With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='potsdam',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='vaihingen256_stride/',
                        help='dataset directory')
    parser.add_argument('--data-list', type=str, default='vaihingen256_stride/test.lst',
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 512], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')

    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--pretrained', type=str,
                        default='pretrain/BAKD/kd_deeplabv3_resnet18_vaihingen_best_model.pth',
                        help='pretrained seg model')
    parser.add_argument('--save-dir', default='../runs/logs/',
                        help='Directory for saving predictions')
    parser.add_argument('--save-pred', action='store_true', default=False,
                        help='save predictions')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='input batch size for training (default: 8)')
    # validation 
    parser.add_argument('--scales', default=[1.], type=float, nargs='+', help='multiple scales')
    parser.add_argument('--flip-eval', action='store_true', default=False,
                        help='flip_evaluation')
    args = parser.parse_args()

    if args.backbone.startswith('resnet'):
        args.aux = True
    elif args.backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args


class Evaluator(object):
    def __init__(self, args, num_gpus):
        self.args = args
        self.num_gpus = num_gpus
        self.device = torch.device(args.device)

        # dataset and dataloader
        if args.dataset == 'citys':
            self.val_dataset = CSValSet(args.data, 'dataset/list/cityscapes111/val.lst', crop_size=(1024, 2048))
        elif args.dataset == 'potsdam':
            self.val_dataset = CSValSet(args.data, 'vaihingen256_stride/test.lst')
        elif args.dataset == 'camvid':
            self.val_dataset = CamvidValSet(args.data, './dataset/list/CamVid/camvid_test_list.txt')

        val_sampler = make_data_sampler(self.val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model,
                                            backbone=args.backbone,
                                            aux=args.aux,
                                            pretrained=args.pretrained,
                                            pretrained_base='None',
                                            local_rank=args.local_rank,
                                            norm_layer=BatchNorm2d,
                                            num_class=self.val_dataset.num_class).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logger.info('Params: %.2fM FLOPs: %.2fG'
                        % (cal_param_size(self.model) / 1e6, cal_multi_adds(self.model, (1, 3, 512, 512)) / 1e9))

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = SegmentationMetric(self.val_dataset.num_class)

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def predict_whole(self, net, image, tile_size):
        interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
        prediction = net(image.cuda())
        if isinstance(prediction, tuple) or isinstance(prediction, list):
            prediction = prediction[0]
        prediction = interp(prediction)
        return prediction

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))

        num_classes = self.val_dataset.num_class
        class_iou = []
        class_f1 = []
        class_oa = []

        total_pixAcc = 0.0
        total_mIoU = 0.0
        total_iou = []
        num_samples = len(self.val_loader)

        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.long().to(self.device)
            N_, C_, H_, W_ = image.size()
            tile_size = (H_, W_)
            full_probs = torch.zeros((1, self.val_dataset.num_class, H_, W_)).cuda()
            scales = args.scales
            with torch.no_grad():
                for scale in scales:
                    scale = float(scale)
                    print("Predicting image scaled by %f" % scale)
                    scale_image = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True)
                    scaled_probs = self.predict_whole(model, scale_image, tile_size)

                    if args.flip_eval:
                        print("flip evaluation")
                        flip_scaled_probs = self.predict_whole(model, torch.flip(scale_image, dims=[3]), tile_size)
                        scaled_probs = 0.5 * (scaled_probs + torch.flip(flip_scaled_probs, dims=[3]))
                    full_probs += scaled_probs
                full_probs /= len(scales)

            self.metric.update(full_probs, target)
            #pixAcc, mIoU = self.metric.get()
            pixAcc, mIoU, iou_list = self.metric.get()
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))
            total_pixAcc += pixAcc
            total_mIoU += mIoU
            #total_iou +=iou_list
            total_iou.append(iou_list)


            if self.args.save_pred:
                pred = torch.argmax(full_probs, 1)
                pred = pred.cpu().data.numpy()
                predict = pred.squeeze(0)
                mask = get_color_pallete(predict, self.args.dataset)
                mask.save(os.path.join(args.outdir, os.path.splitext(filename[0])[0] + '.png'))

            pred = torch.argmax(full_probs, 1)
            #pred = pred.cpu().data.numpy()
            class_iou_per_image = np.zeros((num_classes,), dtype=float)
            class_f1_per_image = np.zeros((num_classes,), dtype=float)
            class_oa_per_image = np.zeros((num_classes,), dtype=float)
            for class_idx in range(num_classes):
                class_mask = (target == class_idx)
                class_pred = (pred == class_idx)
                intersection = np.logical_and(class_pred.cpu().numpy(), class_mask.cpu().numpy())
                union = np.logical_or(class_pred.cpu().numpy(), class_mask.cpu().numpy())

                class_iou_per_image[class_idx] = np.sum(intersection) / (np.sum(union) + 1e-8)

                class_f1_per_image[class_idx] = f1_score(class_mask.cpu().numpy().flatten(),
                                                         class_pred.cpu().numpy().flatten())

                class_oa_per_image[class_idx] = np.count_nonzero(class_pred.cpu() == class_mask.cpu()) / (H_ * W_)

            # 存储每张图片的三个值
            class_iou.append(class_iou_per_image)
            class_f1.append(class_f1_per_image)
            class_oa.append(class_oa_per_image)
        class_iou = np.array(class_iou)  # 转换为NumPy数组
        class_f1 = np.array(class_f1)  # 转换为NumPy数组
        class_oa = np.array(class_oa)  # 转换为NumPy数组

        class_iou_mean = []
        class_f1_mean = []
        class_oa_mean = []
        class_iou_true_mean = []
        num_rows = len(total_iou)
        num_cols = len(total_iou[0])

        column_means = []

        for col in range(num_cols):
            column_values = []

            for row in range(num_rows):
                value = total_iou[row][col]
                if value != 0 and not np.isnan(value):
                    column_values.append(value)

            column_mean = np.mean(column_values)
            column_means.append(column_mean)

        print(column_means)

        for col in range(class_iou.shape[1]):
            col_iou = class_iou[:, col]
            col_f1 = class_f1[:, col]
            col_oa = class_oa[:, col]

            col_iou_mean = np.nanmean(col_iou[col_iou != 0])
            col_f1_mean = np.nanmean(col_f1[col_f1 != 0])
            col_oa_mean = np.nanmean(col_oa[col_oa != 0])


            class_iou_mean.append(col_iou_mean)
            class_f1_mean.append(col_f1_mean)
            class_oa_mean.append(col_oa_mean)


        class_iou_mean = np.array(class_iou_mean)
        class_f1_mean = np.array(class_f1_mean)
        class_oa_mean = np.array(class_oa_mean)

        logger.info("Class-wise IoU:")
        for class_idx in range(num_classes):
            logger.info("Class {}: {:.3f}".format(class_idx, class_iou_mean[class_idx] * 100))


        logger.info("Class-wise IoU_true:")

        for class_idx in range(num_classes):
            logger.info("Class {}: {:.3f}".format(class_idx, column_means[class_idx] * 100))

        logger.info("Class-wise F1:")
        for class_idx in range(num_classes):
            logger.info("Class {}: {:.3f}".format(class_idx, class_f1_mean[class_idx]))

        logger.info("Class-wise OA:")
        for class_idx in range(num_classes):
            logger.info("Class {}: {:.3f}".format(class_idx, class_oa_mean[class_idx] * 100))



        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).cuda().to(args.local_rank)
            sum_total_label = torch.tensor(self.metric.total_label).cuda().to(args.local_rank)
            sum_total_inter = torch.tensor(self.metric.total_inter).cuda().to(args.local_rank)
            sum_total_union = torch.tensor(self.metric.total_union).cuda().to(args.local_rank)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label)  # remove np.spacing(1)
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            # mIoU = IoU.mean().item()
            mIoU = np.nanmean(IoU[:-1])
            total_pixAcc += pixAcc.item()
            total_mIoU += mIoU * num_samples
            total_iou += IoU * num_samples

        total_pixAcc /= num_samples
        total_mIoU /= num_samples
        #total_iou /= num_samples

        logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            total_pixAcc * 100, total_mIoU * 100))

        class_iou_mean = np.mean(class_iou_mean[:num_classes-1])
        logger.info("Class-wise mIoU:")
        logger.info("{:.3f}".format(class_iou_mean * 100))

        class_iou_true_mean = np.mean(column_means[:num_classes-1])
        logger.info("Class-wise mIoU_true:")
        logger.info("{:.3f}".format(class_iou_true_mean * 100))

        class_f1_mean = np.mean(class_f1_mean[:num_classes-1])
        logger.info("Class-wise f1:")
        logger.info("{:.3f}".format(class_f1_mean * 100))

        class_oa_mean = np.mean(class_oa_mean[:num_classes-1])
        logger.info("Class-wise oa:")
        logger.info("{:.3f}".format(class_oa_mean * 100))


        with open('OURS/our_noisy100_potsdam/oure.txt', 'w') as file:
            file.write("Class-wise IoU:\n")
            for class_idx in range(num_classes):
                file.write("Class {}: {:.3f}\n".format(class_idx, class_iou_mean[class_idx] * 100))

            file.write("\nClass-wise IoU_true:\n")
            for class_idx in range(num_classes):
                file.write("Class {}: {:.3f}\n".format(class_idx, column_means[class_idx] * 100))

            file.write("\nClass-wise F1:\n")
            for class_idx in range(num_classes):
                file.write("Class {}: {:.3f}\n".format(class_idx, class_f1_mean[class_idx]))

            file.write("\nClass-wise OA:\n")
            for class_idx in range(num_classes):
                file.write("Class {}: {:.3f}\n".format(class_idx, class_oa_mean[class_idx] * 100))

            file.write("Class-wise mIoU:\n")
            file.write("{:.3f}\n".format(class_iou_mean * 100))

            file.write("\nClass-wise mIoU_true:\n")
            file.write("{:.3f}\n".format(class_iou_true_mean * 100))

            file.write("\nClass-wise f1:\n")
            file.write("{:.3f}\n".format(class_f1_mean * 100))

            file.write("\nClass-wise oa:\n")
            file.write("{:.3f}\n".format(class_oa_mean * 100))
        # 打印结果写入完成
        logger.info("打印结果已写入文件：OUS/ours_potsdam.txt")


        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        # TODO: optim code
    outdir = '{}_{}_{}'.format(args.model, args.backbone, args.dataset)
    args.outdir = os.path.join(args.save_dir, outdir)
    if args.save_pred:
        if (args.distributed and args.local_rank == 0) or args.distributed is False:
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

    logger = setup_logger("semantic_segmentation", args.save_dir, get_rank(),
                          filename='{}_{}_{}_multiscale_val.txt'.format(args.model, args.backbone, args.dataset),
                          mode='a+')

    evaluator = Evaluator(args, num_gpus)
    evaluator.eval()
    torch.cuda.empty_cache()
