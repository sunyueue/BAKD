"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable

__all__ = ['SegCrossEntropyLoss',
           'SegCrossEntropyLoss_weight',
           'CriterionKD', 
           'CriterionMiniBatchCrossImagePair',
           'DIST']


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, z_s, z_t):
        y_s = z_s.softmax(dim=1)
        y_t = z_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss

# TODO: optim function
class SegCrossEntropyLoss_weight(nn.Module):
    def __init__(self, ignore_index=-1, weight=None, **kwargs):
        super(SegCrossEntropyLoss_weight, self).__init__()
        self.task_loss = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=weight, **kwargs)

    def forward(self, inputs, targets):

        B, H, W = targets.size()
        inputs = F.interpolate(inputs, (H, W), mode='bilinear', align_corners=True)
        return self.task_loss(inputs, targets)

class SegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, **kwargs):
        super(SegCrossEntropyLoss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, **kwargs)

    def forward(self, inputs, targets):

        B, H, W = targets.size()
        inputs = F.interpolate(inputs, (H, W), mode='bilinear', align_corners=True)
        return self.task_loss(inputs, targets)

class SegCrossEntropyLoss2(nn.Module):
    def __init__(self, ignore_index=-1, **kwargs):
        super(SegCrossEntropyLoss2, self).__init__()
        self.task_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, **kwargs)

    def forward(self, inputs, targets):


        return self.task_loss(inputs, targets)
class FocalLoss1(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight = None, logits=False, reduce=True):
        super(FocalLoss1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight=weight
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(loss)
        else:
            return loss
class FocalLoss2(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1, epsilon=1.e-9, device=None):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha, device=device)
        else:
            self.alpha = alpha
            self.alpha.to(device)
        self.epsilon = epsilon
        self.num_labels = len(alpha)

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        #
        # # target = target.view(-1)
        # # valid_mask = (target != -1)
        # # target = target[valid_mask]
        # idx = target.long()
        # one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
        # #计算focalloss
        # logits = torch.softmax(input, dim=-1)
        # # 调整 logits 的大小和 targets 一致
        # logits = F.interpolate(logits, size=target.size()[1:], mode='bilinear', align_corners=True)
        # loss = -self.alpha.to(input.device) * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        # loss = loss.sum(dim=1)
        # return loss.mean()
        # target[target == -1] =  5
        # num_labels = input.size(1)
        # idx = target.view(-1, 1).long()
        # one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # # one_hot_key[:, 0] = 0  # ignore 0 index.
        # logits = input.view(-1, input.size(1))
        # logits = torch.softmax(logits, dim=-1)
        # loss = -self.alpha.to("cuda") * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        # loss = loss.sum(1)
        # return loss.mean()

        num_labels = input.size(1)
        valid_targets = target[target != -1]
        idx = valid_targets.view(-1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx.unsqueeze(1), 1)
        # one_hot_key[:, 0] = 0  # ignore 0 index.

        logits = torch.softmax(input.view(-1, num_labels), dim=-1)
        logits = logits.view(input.size(0), -1)
        valid_logits = logits[target != -1]
        loss = -self.alpha.to("cuda") * one_hot_key * torch.pow((1 - valid_logits), self.gamma) * (valid_logits  + self.epsilon).log()
        loss = loss = loss.sum(1).mean()
        return loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=None, ignore_index=-1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha, device=device)
        else:
            self.alpha = alpha
            self.alpha.to(device)
        self.epsilon = epsilon
        self.num_labels = len(alpha)

    def forward(self, input, target):


        num_labels = input.size(1)
        flat_target = target.view(-1)
        valid_targets = target[target != -1]
        idx = valid_targets.view(-1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx.unsqueeze(1), 1)
        # one_hot_key[:, 0] = 0  # ignore 0 index.

        logits = torch.softmax(input.view(-1, num_labels), dim=-1)
        # 使用 gather 操作提取有效的 logits
        valid_logits = logits[flat_target != -1, :]
        valid_logits = valid_logits.gather(1, idx.unsqueeze(1))
        loss = -self.alpha.to("cuda") * one_hot_key * torch.pow((1 - valid_logits), self.gamma) * (valid_logits  + self.epsilon).log()
        loss = loss.sum(1).mean()
        return loss



class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=4, reduction='batchmean'):
        super(CriterionKD, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, pred, soft):
        if self.reduction == 'none':
            scale_pred = pred.permute(0, 2, 3, 1).contiguous()
            #print("scale_pred")
            #print(scale_pred)
            scale_soft = soft.permute(0, 2, 3, 1).contiguous()
            p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
            #print("p_s")
            #print(p_s)
            p_t = F.softmax(scale_soft / self.temperature, dim=1)
            loss = F.kl_div(p_s, p_t, reduction=self.reduction) * (self.temperature ** 2)
        else:
            B, C, h, w = soft.size()
            scale_pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
            scale_soft = soft.permute(0, 2, 3, 1).contiguous().view(-1, C)
            p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
            p_t = F.softmax(scale_soft / self.temperature, dim=1)
            loss = F.kl_div(p_s, p_t, reduction=self.reduction) * (self.temperature ** 2)
        return loss


# class CriterionKD(nn.Module):
#     '''
#     knowledge distillation loss
#     '''
#     def __init__(self, temperature=4,
#                  reduction='batchmean'):
#         super(CriterionKD, self).__init__()
#         self.temperature = temperature
#         self.reduction = reduction
#
#     def forward(self, pred, soft):
#         B, C, h, w = soft.size()
#         scale_pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
#         scale_soft = soft.permute(0, 2, 3, 1).contiguous().view(-1, C)
#         p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
#         p_t = F.softmax(scale_soft / self.temperature, dim=1)
#         loss = F.kl_div(p_s, p_t, reduction=self.reduction) * (self.temperature ** 2)
#         return loss


# def Std(pred):
#     #pred 模型输出:[8,19,64,128],计算std,要明白我们是否要计算的是[B,H,W]的平方
#     [B,K,H,W]=pred.size()
#     for i in range(B):
#         for j in range(K):
#             for k in range(H):
#                 for m in range(W):
#                     pred[i][j][k][m]**2
#             #v_i = torch.bmm(pred[i],pred[i]) jisuan pingfang
#         #jisuan 1/K
#         #pinjie
#     #output:[B]

class CriterionKD1(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=1, reduction='batchmean'):
        super(CriterionKD1, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, pred, soft):
        if self.reduction == 'none':
            scale_pred = pred.permute(0, 2, 3, 1).contiguous()
            scale_soft = soft.permute(0, 2, 3, 1).contiguous()
            p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
            p_t = F.softmax(scale_soft / self.temperature, dim=1)
            loss = F.kl_div(p_s, p_t, reduction=self.reduction) * (self.temperature ** 2)
        else:

            B, C, h, w = soft.size()
            scale_pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
            scale_soft = soft.permute(0, 2, 3, 1).contiguous().view(-1, C)
            # print("scale_pred")
            # print(scale_soft)
            # print("scale_pred")
            # print(scale_soft)


            #K = pred.size()[1]
            TT_s = torch.std(scale_pred,dim=1)
            TT_t = torch.std(scale_soft,dim=1)
            T_s = torch.reshape(TT_s,(-1,1))

            T_t = torch.reshape(TT_t, (-1,1))
            # print("T_t")
            # print(T_t)
            mean_s=torch.mean(TT_s)
            mean_t=torch.mean(TT_t)
            # print("mean_s")
            # print(mean_s)
            # print("mean_t")
            # print(mean_t)
            p_s = F.log_softmax(scale_pred / T_s, dim=1)
            p_t = F.softmax(scale_soft / T_t, dim=1)
            #loss = F.kl_div(p_s, p_t, reduction=self.reduction) * (self.temperature ** 2)
            T=torch.tensor(0.3)
            loss = (F.kl_div(p_s, p_t, reduction=self.reduction))
            print("loss")
            print(loss)

            with open(r'/home/cver/data/sy/DCS/T_s.txt','a') as f:
                 f.write(str(mean_s))
            with open(r'/home/cver/data/sy/DCS/T_t.txt','a') as f:
                f.write(str(mean_t))
        return loss












class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CriterionMiniBatchCrossImagePair(nn.Module):
    def __init__(self, temperature):
        super(CriterionMiniBatchCrossImagePair, self).__init__()
        self.temperature = temperature

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
        
        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output


    def forward(self, feat_S, feat_T):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        B, C, H, W = feat_S.size()

        '''
        patch_w = 2
        patch_h = 2
        #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        '''
        
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)
        
        sim_dis = torch.tensor(0.).cuda()
        for i in range(B):
            for j in range(B):
                s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[j])
                t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[j])

                p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
                p_t = F.softmax(t_sim_map / self.temperature, dim=1)

                sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
                sim_dis += sim_dis_
        sim_dis = sim_dis / (B * B)
        return sim_dis
