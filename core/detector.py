import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list
from .selector import filter_submod_selection

import os
import os.path as osp
import numpy as np

__all__ = ["RandBox"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
                [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
            ).reshape(-1, 2)  # n*4,2
            
def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)  # n,4
    else:
        return point.new_zeros(0, 4)

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)
    
def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def Revision_PRED(Instance1, Instance2):
    '''
    Based on Instance1 to revise Instance2,
    we revise Instance2 by scores and ious when both of Instances have scores,
    Namely,(Instance1[pred_output1],Instance2[pred_output2]),depending on their scores and ious

    '''
    assert Instance1.image_size == Instance2.image_size
    image_size = Instance1.image_size

    refine_gt_Instance = Instances(tuple(image_size))
    missing_Instance = Instances(tuple(image_size))

    bboxes1 = Instance1.pred_boxes.tensor
    scores1 = Instance1.scores
    classes1 = Instance1.pred_classes

    bboxes2 = Instance2.pred_boxes.tensor.clone()
    scores2 = Instance2.scores.clone()
    classes2 = Instance2.pred_classes.clone()

    ious = bbox_overlaps(bboxes1, bboxes2)

    if len(ious)==0 or len(ious[0])==0:
        return Instances.cat([Instance1,Instance2])

    while(True):

        refine_gt_inds = (ious > 0.5).any(dim=0)
        refine_inds = ious.max(dim=0)[1]

        refine_pred_scores = scores1[refine_inds]
        need_refine = refine_pred_scores >= scores2

        lower_scores_inds = (~refine_gt_inds | ~need_refine) & refine_gt_inds

        lower_scores_inds0 = torch.where(lower_scores_inds)[0]

        if lower_scores_inds0.numel()>0:
            index = [refine_inds[lower_scores_inds],lower_scores_inds0]

            input_zeros = torch.zeros((lower_scores_inds0.numel())).to(ious.device) + 0.5

            ious.index_put_(index,input_zeros)

        else:
            break

    refine_inds = refine_inds[refine_gt_inds]
    refine_gt_inds = torch.where(refine_gt_inds)[0]
    refine_gt_inds_repeat = refine_gt_inds.reshape(-1,1).repeat(1,4)

    bboxes2.scatter_(dim=0,index=refine_gt_inds_repeat,src=bboxes1[refine_inds])
    classes2.scatter_(dim=0,index=refine_gt_inds,src=classes1[refine_inds])
    scores2.scatter_(dim=0,index=refine_gt_inds,src=scores1[refine_inds])

    refine_gt_Instance.pred_boxes = Boxes(bboxes2)
    refine_gt_Instance.pred_boxes.clip(image_size)
    refine_gt_Instance.pred_classes = classes2
    refine_gt_Instance.scores = scores2


    missing_inds = (ious<0.5).all(dim=1)
    missing_Instance.pred_boxes = Boxes(bboxes1[missing_inds])
    missing_Instance.pred_classes = classes1[missing_inds]
    missing_Instance.scores = scores1[missing_inds]

    return Instances.cat([missing_Instance,refine_gt_Instance])

def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()
    area2 = boxes2.area()

    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoA (intersection over area)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoA, sized [N,M].
    """
    area1 = boxes1.area()
    area2 = boxes2.area()

    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]   # [N,M]
    del width_height

    # handle empty boxes
    ioa = torch.where(
        inter > 0,
        inter / (area1[:, None]),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return ioa

@META_ARCH_REGISTRY.register()
class RandBox(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.NUM_CLASSES     # 80 classes + 1 unknown + 1 bg
        self.num_proposals = cfg.MODEL.NUM_PROPOSALS # This is 500 in OrthogonalDet
        self.hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.num_heads = cfg.MODEL.NUM_HEADS
        self.sampling_method = cfg.MODEL.SAMPLING_METHOD
        self.disentangled = cfg.MODEL.DISENTANGLED  
        self.score_threshold = cfg.MODEL.SCORE_THRESHOLD
        self.gt_iou_threshold = cfg.MODEL.GT_IOU_THRESHOLD
        self.known_obj_threshold = cfg.MODEL.DSTG_OBJ_THRESHOLD
        self.sim_threshold = cfg.MODEL.D2TG_SIM_THRESHOLD
        self.obj_threshold = cfg.MODEL.D2TG_OBJ_THRESHOLD
        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Selection Config - Currently run offline
        self.selection = cfg.DISCOVER_UNKNOWN
        self.selection_function_name = cfg.DISCOVER_FUNCTION_NAME

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.multiple_sample = cfg.MODEL.M_STEP
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.SNR_SCALE

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.x_dic = torch.rand((10000, 4))
        self.x_meta = torch.arange(start=-2, end=2, step=0.4)
        for i1 in range(10):
            for i2 in range(10):
                for i3 in range(10):
                    for i4 in range(10):
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][0], \
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][1], \
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][2], \
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][3] = self.x_meta[i1], self.x_meta[i2], \
                        self.x_meta[i3], self.x_meta[i4]
        self.x_dic = self.x_dic[torch.randperm(self.x_dic.size(0))]
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        # Loss parameters:
        class_weight = cfg.MODEL.CLASS_WEIGHT
        giou_weight = cfg.MODEL.GIOU_WEIGHT
        l1_weight = cfg.MODEL.L1_WEIGHT
        nc_weight = cfg.MODEL.NC_WEIGHT
        no_object_weight = cfg.MODEL.NO_OBJECT_WEIGHT
        decorr_weight = cfg.MODEL.DECORR_WEIGHT
        crowd_weight = cfg.MODEL.CROWD_WEIGHT
        self.deep_supervision = cfg.MODEL.DEEP_SUPERVISION
        self.use_nms = cfg.MODEL.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight,
                       "loss_nc_ce": nc_weight, "loss_decorr": decorr_weight, "loss_crowd": crowd_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]
        if cfg.MODEL.NC:
            losses += ["nc_labels"]
        if cfg.MODEL.CROWD and crowd_weight > 0:
            losses += ["crowd"]
        if decorr_weight > 0:
            losses += ["decorr"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def freeze_all(self):
        for module in self.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                value.requires_grad = False

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False, sample_i=0):
        if self.sampling_method == 'Random':
            x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x_boxes = ((x_boxes / self.scale) + 1) / 2
        else:
            x_boxes = self.x_dic.to(x.device)[self.num_proposals * sample_i:self.num_proposals * (sample_i + 1), :]
            x_boxes = ((x_boxes / self.scale) + 1) / 2

        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_objectness, outputs_coord,_ = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_objectness, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        x_start = None
        if self.sampling_method == 'Random':
            for time, time_next in time_pairs:
                time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
                self_cond = x_start if self.self_condition else None

                preds, class_cat, objectness_cat, coord_cat = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                     self_cond, clip_x_start=clip_denoised)
                pred_noise, x_start = preds.pred_noise, preds.pred_x_start
        else:
            for sample_step in range(self.multiple_sample):
                for time, time_next in time_pairs:
                    time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
                    self_cond = x_start if self.self_condition else None

                    preds, outputs_class, outputs_objectness, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img,
                                                                                 time_cond,
                                                                                 self_cond, clip_x_start=clip_denoised,
                                                                                 sample_i=sample_step)
                if sample_step == 0:
                    class_cat = outputs_class
                    objectness_cat = outputs_objectness
                    coord_cat = outputs_coord
                else:
                    class_cat = torch.cat((class_cat, outputs_class), 2)
                    objectness_cat = torch.cat((objectness_cat, outputs_objectness), 2)
                    coord_cat = torch.cat((coord_cat, outputs_coord), 2)

        results = self.inference(class_cat[-1], objectness_cat[-1], coord_cat[-1], images.image_sizes)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def mine_unknown_rois(self, batched_inputs, backbone_feats, images_whwh, images):
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets, x_boxes, t = self.prepare_targets(gt_instances)
        t = t.squeeze(-1)
        x_boxes = x_boxes * images_whwh[:, None, :]
        
        outputs_class, output_objectness, outputs_coord, proposal_features = self.head(backbone_feats, 
                                                                                       x_boxes, 
                                                                                       t, 
                                                                                       None, 
                                                                                       roi_labels=None)
        raw_rois = {
            'pred_logits': outputs_class[-1], 
            'pred_objectness': output_objectness[-1], 
            'pred_boxes': outputs_coord[-1],
            'pred_proposal_features': proposal_features[-1]
        }
        
        # Map each RoI to its corresponding class label using the Dynamic Matcher
        # Note: We only look for the knowns here, remaining RoIs are marked as unknown
        gt_indices, _, _, _ = self.criterion.matcher(raw_rois, targets)   
        
        raw_rois_list = []
        gt_indices_list = []        
        raw_scores_list = []
        raw_bbox_list = []
        for batch_idx in range(len(targets)):
            gt_indices_list.append(gt_indices[batch_idx][0].float()) 
            raw_rois_list.append(raw_rois["pred_proposal_features"][batch_idx])
            raw_scores_list.append(raw_rois["pred_objectness"][batch_idx])
            raw_bbox_list.append(raw_rois["pred_boxes"][batch_idx])
        
        # flatten the feature vector across images in the batch
        raw_rois = torch.cat(raw_rois_list, dim = 0)
        known_mask = torch.cat(gt_indices_list, dim = 0)
        raw_objectness = torch.cat(raw_scores_list, dim = 0)
        raw_bboxes = torch.cat(raw_bbox_list, dim = 0)
        
        # Remove low score RoIs - Not important and mostly would contain BG objects
        low_score_mask = (raw_objectness.squeeze(1) > 0.2)
        low_score_filtered_idx = torch.nonzero(low_score_mask, as_tuple=False).squeeze(1)
        
        # Apply the low score filter
        raw_rois = raw_rois[low_score_filtered_idx]
        known_mask = known_mask[low_score_filtered_idx]
        raw_objectness = raw_objectness[low_score_filtered_idx]
        raw_bboxes = raw_bboxes[low_score_filtered_idx]
               
        unk_idx, outputs_coord, output_objectness = filter_submod_selection(
                                                                    self.selection_function_name,
                                                                    known_mask, 
                                                                    raw_bboxes, 
                                                                    raw_rois, 
                                                                    outputs_class, 
                                                                    raw_objectness,
                                                                    10
                                                                )
        
        # Remove RoIs with objectness < 0.4 - We need only important RoIs
        final_filter_mask = output_objectness.squeeze(-1) >= 0.4
        filtered_idx = torch.nonzero(final_filter_mask, as_tuple=False).squeeze(1)

        # If no RoIs pass the filter, return empty tensors
        if filtered_idx.numel() == 0:
            return {
                'bboxes': torch.empty((0, 4), device=self.device),
                'labels': torch.empty((0,), device=self.device, dtype=torch.long),
                'scores': torch.empty((0,), device=self.device)
            }

        # Apply the final filter
        outputs_coord = outputs_coord[filtered_idx]
        unk_idx = unk_idx[filtered_idx]
        del output_objectness
        
        return {
            'bboxes': outputs_coord,
            'labels': torch.ones_like(unk_idx) * 80, 
        }

    def cvt_Instance_list(self, source_TM, source_Instance, target_TM, target_img_shapes):
        assert len(source_TM) == len(source_Instance) == len(target_TM) == len(target_img_shapes)
        target_Instances = []
        if source_Instance[0].has('pred_boxes'):
            bbox_key = 'pred_boxes'
            class_key = 'pred_classes'
        else:
            bbox_key = 'gt_boxes'
            class_key = 'gt_classes' 

        for i, (instance, source_tm, target_tm, target_img_shape) in enumerate(zip(source_Instance, source_TM, target_TM, target_img_shapes)):
            source_boxes = instance.get(bbox_key).tensor
            
            if len(source_boxes) != 0:
                source_points = bbox2points(source_boxes[:, :4])
                source_points = torch.cat(
                            [source_points, source_points.new_ones(source_points.shape[0], 1)], dim=1
                        )
                M_T = np.matmul(target_tm, np.linalg.inv(source_tm))
                M_T = torch.tensor(M_T).to(source_boxes.device).float()
                target_points = (M_T @ source_points.t()).t()
                H, W = target_img_shape
                target_points = target_points[:, :2] / target_points[:, 2:3]
                target_bboxes = points2bbox(target_points, W, H)
                minx, min_y, max_x, max_y = target_bboxes[:, 0], target_bboxes[:, 1], target_bboxes[:, 2], target_bboxes[:, 3]
                valid_bboxes = (minx < max_x) & (min_y < max_y)

                target = Instances(target_img_shape)
                target.set(bbox_key, Boxes(target_bboxes[valid_bboxes]))
                if instance.has('scores'):
                    target.scores = instance.scores[valid_bboxes]
                target.set(class_key, instance.get(class_key)[valid_bboxes])
            else:
                target = Instances(target_img_shape)
                target.set(bbox_key, Boxes(torch.zeros(0, 4, device=self.device)))
                if instance.has('scores'):
                    target.scores = torch.zeros(0, device=self.device)
                target.set(class_key, torch.zeros(0, dtype=torch.long, device=self.device))
    
            target_Instances.append(target)

        return target_Instances
    
    def merge_ground_truth(self, targets, predictions, images, iou_thresold, source_TM, target_TM):
        new_targets = []
        known_pseudolabels = []

        for targets_per_image, predictions_per_image, image, source_TM_i, target_TM_i in zip(targets, predictions, images, source_TM, target_TM):
            image_size = image.shape[1:3]

            if len(predictions_per_image.pred_boxes) != 0:
                if predictions_per_image.has("indices"):
                    predictions_per_image_cvt = self.cvt_bbox(source_TM_i,
                                                                predictions_per_image.pred_boxes.tensor,
                                                                target_TM_i,
                                                                image.shape,
                                                                predictions_per_image.pred_classes,
                                                                predictions_per_image.scores,
                                                                predictions_per_image.indices
                                                                )
                else:
                    predictions_per_image_cvt = self.cvt_bbox(source_TM_i,
                                                                predictions_per_image.pred_boxes.tensor,
                                                                target_TM_i,
                                                                image.shape,
                                                                predictions_per_image.pred_classes,
                                                                predictions_per_image.scores,
                                                                )
            else:
                predictions_per_image_cvt = Instances(image_size)
                predictions_per_image_cvt.pred_boxes = Boxes(torch.zeros(0, 4, device=self.device))
                predictions_per_image_cvt.scores = torch.zeros(0, device=self.device)
                predictions_per_image_cvt.pred_classes = torch.zeros(0, dtype=torch.long, device=self.device)
                predictions_per_image_cvt.indices = torch.full(
                    (len(predictions_per_image_cvt),),
                    9999,
                    device=predictions_per_image.scores.device,
                    dtype=torch.long
                )

            iou_matrix = pairwise_iou(targets_per_image.gt_boxes,
                                      predictions_per_image_cvt.pred_boxes)
            # ioa_matrix = pairwise_ioa(targets_per_image.gt_boxes,
            #                           predictions_per_image_cvt.pred_boxes)
            iou_filter = iou_matrix > iou_thresold
            # ioa_filter = ioa_matrix > 0.99

            target_class_list = (targets_per_image.gt_classes).reshape(-1, 1)
            pred_class_list = (predictions_per_image_cvt.pred_classes).reshape(1, -1)
            class_filter = target_class_list == pred_class_list

            final_filter = iou_filter & class_filter
            # final_filter = (iou_filter & class_filter) | (ioa_filter & class_filter)
            unlabel_idxs = torch.sum(final_filter, 0) == 0

            new_target = Instances(image_size)
            new_target.gt_boxes = Boxes.cat([targets_per_image.gt_boxes,
                                             predictions_per_image_cvt.pred_boxes[unlabel_idxs]])
            new_target.gt_classes = torch.cat([targets_per_image.gt_classes,
                                               predictions_per_image_cvt.pred_classes[unlabel_idxs]])
            new_targets.append(new_target)
            
            pred_new = Instances(image_size)
            pred_new.pred_boxes = predictions_per_image_cvt.pred_boxes[unlabel_idxs]
            pred_new.pred_classes = predictions_per_image_cvt.pred_classes[unlabel_idxs]
            if predictions_per_image_cvt.has("indices"):
                pred_new.indices = predictions_per_image_cvt.indices[unlabel_idxs]

            if hasattr(predictions_per_image_cvt, "scores"):
                pred_new.scores = predictions_per_image_cvt.scores[unlabel_idxs]

            known_pseudolabels.append(pred_new)            

        return new_targets, known_pseudolabels

    def cvt_bbox(self, source_TM, source_bbox, target_TM, target_img_shape, labels, scores, indices=None):

        source_points = bbox2points(source_bbox[:, :4])
        source_points = torch.cat(
                    [source_points, source_points.new_ones(source_points.shape[0], 1)], dim=1
                )
        M_T = np.matmul(target_TM, np.linalg.inv(source_TM))
        M_T = torch.tensor(M_T).to(source_bbox.device).float()
        target_points = (M_T @ source_points.t()).t()
        _, H, W = target_img_shape
        target_points = target_points[:, :2] / target_points[:, 2:3]
        target_bboxes = points2bbox(target_points, W, H)
        minx, min_y, max_x, max_y = target_bboxes[:, 0], target_bboxes[:, 1], target_bboxes[:, 2], target_bboxes[:, 3]
        valid_bboxes = (minx < max_x) & (min_y < max_y)

        target = Instances(target_img_shape[1:3])
        target.pred_boxes = Boxes(target_bboxes[valid_bboxes])
        target.scores = scores[valid_bboxes]
        target.pred_classes = labels[valid_bboxes]
        if indices is not None:
            target.indices = indices[valid_bboxes]
        return target

    def create_instances_from_outputs(self, logits, boxes, objectness, image_size, score_threshold, obj_threshold):
        topk_candidates = 10
                
        probs = F.softmax(logits[:, :-1], dim=-1)
        scores, classes = torch.max(probs, dim=-1)
        
        # 🔹 objectness threshold 적용
        objectness = objectness.squeeze(-1)
        obj_keep = objectness >= obj_threshold
        scores = scores[obj_keep]
        classes = classes[obj_keep]
        boxes = boxes[obj_keep]
        objectness = objectness[obj_keep]
        
        num_topk = min(topk_candidates, logits.size(0))
        topk_scores, topk_idxs = scores.sort(descending=True)
        topk_scores = topk_scores[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        keep_idxs = topk_scores >= score_threshold
        final_scores = topk_scores[keep_idxs]
        final_boxes = boxes[topk_idxs[keep_idxs]]
        final_classes = classes[topk_idxs[keep_idxs]]
        
        keep = batched_nms(final_boxes, final_scores, final_classes, iou_threshold=0.6)
        keep = keep[:100]
        
        instance = Instances(image_size)
        instance.pred_boxes = Boxes(final_boxes[keep])
        instance.scores = final_scores[keep]
        instance.pred_classes = final_classes[keep]

        return instance        

    def create_unknown_instances_from_outputs(self, logits, boxes, objectness, image_size):
        probs = torch.softmax(logits[:, :-1], dim=-1)
        # probs = torch.softmax(logits[:, :-1], dim=-1) * objectness
        scores, classes = torch.max(probs, dim=-1)
        objectness = objectness.squeeze(1)
        
        instance = Instances(image_size)
        instance.pred_boxes = Boxes(boxes)
        instance.scores = scores
        instance.objectness = objectness
        instance.pred_classes = classes
        instance.indices = torch.arange(logits.size(0), device=logits.device)

        return instance    

    def cvt_bbox_onlybbox(self, source_TM, source_bbox, target_TM, target_img_shape):

        source_points = bbox2points(source_bbox[:, :4])
        source_points = torch.cat(
                    [source_points, source_points.new_ones(source_points.shape[0], 1)], dim=1
                )
        M_T = np.matmul(target_TM, np.linalg.inv(source_TM))
        M_T = torch.tensor(M_T).to(source_bbox.device).float()
        target_points = (M_T @ source_points.t()).t()
        _, H, W = target_img_shape
        target_points = target_points[:, :2] / target_points[:, 2:3]
        target_bboxes = points2bbox(target_points, W, H)
        minx, min_y, max_x, max_y = target_bboxes[:, 0], target_bboxes[:, 1], target_bboxes[:, 2], target_bboxes[:, 3]
        valid_bboxes = (minx < max_x) & (min_y < max_y)
        target_bboxes = target_bboxes.clone()
        target_bboxes[~valid_bboxes] = 0.0

        return target_bboxes

    def get_cosine_disagreement_instances(
        self,
        pred_logits1,
        pred_logits2,
        preds1,
        preds2,
        gt_instances,
        image_size,
        cos_thr,
        obj_thr,
        gt_iou_thr,
    ):
        # 1. softmax
        prob1 = F.softmax(pred_logits1, dim=1)
        prob2 = F.softmax(pred_logits2, dim=1)

        # 2. cosine similarity filter
        cos_sim = F.cosine_similarity(prob1, prob2, dim=1) # (prob1 * prob2).sum(dim=1) / (torch.norm(prob1, dim=1) * torch.norm(prob2, dim=1) + 1e-8)
        cosine_idxs = torch.where(cos_sim <= cos_thr)[0]

        if len(cosine_idxs) == 0:
            return Instances(image_size)
        
        # 3. objectness filter
        objectness1 = preds1.objectness[cosine_idxs]
        objectness2 = preds2.objectness[cosine_idxs]
        objectness_filter = (objectness1 >= obj_thr) & (objectness2 >= obj_thr)
        objectness_idxs = cosine_idxs[objectness_filter]
        score_idxs = objectness_idxs

        if len(score_idxs) == 0:
            return Instances(image_size)

        # 4. class mismatch filter
        classes1 = preds1.pred_classes[score_idxs]
        classes2 = preds2.pred_classes[score_idxs]
        class_mismatch_filter = classes1 != classes2
        class_mismatch_idxs = score_idxs[class_mismatch_filter]

        if len(class_mismatch_idxs) == 0:
            return Instances(image_size)
        
        # 5. geometry filter
        boxes = preds1.pred_boxes[class_mismatch_idxs].tensor
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        geometry_filter = (x2 > x1) & (y2 > y1) & (area > 1.0)
        geometry_idxs = class_mismatch_idxs[geometry_filter]

        if len(geometry_idxs) == 0:
            return Instances(image_size)

        # 6. GT overlap filter
        if len(gt_instances) > 0:
            ious = pairwise_iou(gt_instances.gt_boxes, preds1.pred_boxes[geometry_idxs])  # (N_gt, N_pred)
            
            iou_filter = ious > gt_iou_thr
            
            gt_class_list = (gt_instances.gt_classes).reshape(-1, 1)
            pred_class_list = (preds1.pred_classes[geometry_idxs]).reshape(1, -1)
            class_filter = gt_class_list == pred_class_list

            known_filter = (iou_filter & class_filter)
            known_idxs = torch.sum(known_filter, 0) == 0
            unknown_idxs = geometry_idxs[known_idxs]
        else:
            unknown_idxs = geometry_idxs

        if len(unknown_idxs) == 0:
            return Instances(image_size)
        
        final_scores = preds1.scores[unknown_idxs]
        final_boxes = preds1.pred_boxes[unknown_idxs].tensor  # Boxes 객체에서 텐서로 변환
        final_classes = preds1.pred_classes[unknown_idxs]

        # 7. NMS
        keep = batched_nms(final_boxes, final_scores, final_classes, iou_threshold=0.6)
        keep = keep[:500]
        
        # 8. Instances
        inst = Instances(image_size)

        inst.pred_boxes   = preds1.pred_boxes[unknown_idxs[keep]]
        inst.pred_classes = preds1.pred_classes[unknown_idxs[keep]]
        inst.scores       = preds1.scores[unknown_idxs[keep]]
        inst.indices      = preds1.indices[unknown_idxs[keep]]

        inst.pred_boxes2   = preds2.pred_boxes[unknown_idxs[keep]]
        inst.pred_classes2 = preds2.pred_classes[unknown_idxs[keep]]
        inst.scores2       = preds2.scores[unknown_idxs[keep]]
        inst.indices2      = preds2.indices[unknown_idxs[keep]]

        inst.cos_sim = cos_sim[unknown_idxs[keep]]

        return inst 
    
    def forward(self, batched_inputs, teacher_model=None, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images_raw, images_nor, images_str, images_raw_whwh, images_nor_whwh, images_str_whwh = self.preprocess_image(batched_inputs)

        #########################################################################

        if not self.training:
            if isinstance(images_raw, (list, torch.Tensor)):
                images_raw = nested_tensor_from_tensor_list(images_raw)

            # Feature Extraction.
            src = self.backbone(images_raw.tensor)
            features = list()
            for f in self.in_features:
                feature = src[f]
                features.append(feature)

            if not self.selection:
                results = self.ddim_sample(batched_inputs, features, images_raw_whwh, images_raw, do_postprocess=do_postprocess)
                return results
            
            if self.selection:
                results = self.mine_unknown_rois(batched_inputs, features, images_raw_whwh, images_raw)
                return results
        
        #########################################################################

        if self.training:
            if isinstance(images_raw, (list, torch.Tensor)):
                images_raw = nested_tensor_from_tensor_list(images_raw)
                
            if isinstance(images_nor, (list, torch.Tensor)):
                images_nor = nested_tensor_from_tensor_list(images_nor)
            
            if isinstance(images_str, (list, torch.Tensor)):
                images_str = nested_tensor_from_tensor_list(images_str)

            with torch.no_grad():
                src_raw = teacher_model.backbone(images_raw.tensor)
            src_nor = self.backbone(images_nor.tensor)
            src_str = self.backbone(images_str.tensor)
            features_raw = list()
            features_nor = list()
            features_str = list()
            for f in self.in_features:
                feature_raw = src_raw[f]
                feature_nor = src_nor[f]
                feature_str = src_str[f]
                features_raw.append(feature_raw)
                features_nor.append(feature_nor)
                features_str.append(feature_str)


        if self.training:
            gt_instances_raw = [x["instances_raw"].to(self.device) for x in batched_inputs]
            gt_instances_nor = [x["instances_nor"].to(self.device) for x in batched_inputs]
            gt_instances_str = [x["instances_str"].to(self.device) for x in batched_inputs]
            TM_raw = [x["image_raw_matrix"] for x in batched_inputs]
            TM_nor = [x["image_nor_matrix"] for x in batched_inputs]
            TM_str = [x["image_str_matrix"] for x in batched_inputs]
            _, x_boxes_raw, t_raw = self.prepare_targets(gt_instances_raw)
            _, x_boxes_nor, t_nor = self.prepare_targets(gt_instances_nor)
            _, x_boxes_str, t_str = self.prepare_targets(gt_instances_str)
            t_raw = t_raw.squeeze(-1)
            t_nor = t_nor.squeeze(-1)
            t_str = t_str.squeeze(-1)
            x_boxes_raw = x_boxes_raw * images_raw_whwh[:, None, :]
            x_boxes_nor = x_boxes_nor * images_nor_whwh[:, None, :]
            x_boxes_str = x_boxes_str * images_str_whwh[:, None, :]


            with torch.no_grad():
                preds_class_raw, preds_objectness_raw, preds_coord_raw, proposal_features_raw = teacher_model.head(features_raw, x_boxes_raw, t_raw, None, roi_labels=None)
            preds_class_nor, preds_objectness_nor, preds_coord_nor, proposal_features_nor = self.head(features_nor, x_boxes_nor, t_nor, None, roi_labels=None)
            preds_class_str, preds_objectness_str, preds_coord_str, proposal_features_str = self.head(features_str, x_boxes_str, t_str, None, roi_labels=None)
                      
            output_raw = {
                'pred_logits': preds_class_raw[-1], 
                'pred_objectness': preds_objectness_raw[-1], 
                'pred_boxes': preds_coord_raw[-1],
                'pred_proposal_features': proposal_features_raw[-1]
            }
            output_nor = {
                'pred_logits': preds_class_nor[-1],
                'pred_objectness': preds_objectness_nor[-1],
                'pred_boxes': preds_coord_nor[-1],
                'pred_proposal_features': proposal_features_nor[-1]
            }
            output_str = {
                'pred_logits': preds_class_str[-1],
                'pred_objectness': preds_objectness_str[-1],
                'pred_boxes': preds_coord_str[-1],
                'pred_proposal_features': proposal_features_str[-1]
            }
                      
            if self.deep_supervision:
                output_raw['aux_outputs'] = [{'pred_logits': a, 'pred_objectness': b, 'pred_boxes': c, 'pred_proposal_features': d}
                                         for a, b, c, d in zip(preds_class_raw[:-1], 
                                                               preds_objectness_raw[:-1], 
                                                               preds_coord_raw[:-1], 
                                                               proposal_features_raw[:-1])]
                output_nor['aux_outputs'] = [{'pred_logits': a, 'pred_objectness': b, 'pred_boxes': c, 'pred_proposal_features': d}
                                         for a, b, c, d in zip(preds_class_nor[:-1], 
                                                               preds_objectness_nor[:-1], 
                                                               preds_coord_nor[:-1], 
                                                               proposal_features_nor[:-1])]
                output_str['aux_outputs'] = [{'pred_logits': a, 'pred_objectness': b, 'pred_boxes': c, 'pred_proposal_features': d}
                                         for a, b, c, d in zip(preds_class_str[:-1], 
                                                               preds_objectness_str[:-1], 
                                                               preds_coord_str[:-1], 
                                                               proposal_features_str[:-1])]
                
            #########################################################################
            
            preds_raw = [
                self.create_instances_from_outputs(preds_class_raw[-1][i], preds_coord_raw[-1][i], preds_objectness_raw[-1][i], images_raw.image_sizes[i], self.score_threshold, self.known_obj_threshold)
                for i in range(len(batched_inputs))
            ]
            preds_nor = [
                self.create_instances_from_outputs(preds_class_nor[-1][i], preds_coord_nor[-1][i], preds_objectness_nor[-1][i], images_nor.image_sizes[i], self.score_threshold, self.known_obj_threshold)
                for i in range(len(batched_inputs))
            ]
            preds_str = [
                self.create_instances_from_outputs(preds_class_str[-1][i], preds_coord_str[-1][i], preds_objectness_str[-1][i], images_str.image_sizes[i], self.score_threshold, self.known_obj_threshold)
                for i in range(len(batched_inputs))
            ]

            preds_raw_to_nor = self.cvt_Instance_list(TM_raw, preds_raw, TM_nor, images_nor.image_sizes)
            preds_raw_to_str = self.cvt_Instance_list(TM_raw, preds_raw, TM_str, images_str.image_sizes)

            preds_nor_new = [Revision_PRED(pred_raw_to_nor, pred_nor) for pred_raw_to_nor, pred_nor in zip(preds_raw_to_nor, preds_nor)]
            preds_str_new = [Revision_PRED(pred_raw_to_str, pred_str) for pred_raw_to_str, pred_str in zip(preds_raw_to_str, preds_str)]

            gt_instances_nor_new, known_pseudolabel_nor = self.merge_ground_truth(gt_instances_nor, preds_str_new, images_nor, self.gt_iou_threshold, TM_str, TM_nor)
            gt_instances_str_new, known_pseudolabel_str = self.merge_ground_truth(gt_instances_str, preds_nor_new, images_str, self.gt_iou_threshold, TM_nor, TM_str)

            targets_nor_new, _, _ = self.prepare_targets(gt_instances_nor_new)
            targets_str_new, _, _ = self.prepare_targets(gt_instances_str_new)

            #########################################################################
            
            bbox_str_from_nor = torch.stack([self.cvt_bbox_onlybbox(TM_nor[i], preds_coord_nor[-1][i], TM_str[i], images_str[i].shape)
                                           for i in range(preds_class_nor[-1].shape[0])], dim=0)
            preds_class_str_from_nor, preds_objectness_str_from_nor, preds_coord_str_from_nor, _ = self.head(features_str, bbox_str_from_nor, t_str, None, roi_labels=None, samerpn=True)              
            
            bbox_nor_from_str = torch.stack([self.cvt_bbox_onlybbox(TM_str[i], preds_coord_str[-1][i], TM_nor[i], images_nor[i].shape)
                                           for i in range(preds_class_str[-1].shape[0])], dim=0)
            preds_class_nor_from_str, preds_objectness_nor_from_str, preds_coord_nor_from_str, _ = self.head(features_nor, bbox_nor_from_str, t_nor, None, roi_labels=None, samerpn=True)

            preds_nor_all = [
                self.create_unknown_instances_from_outputs(preds_class_nor[-1][i], preds_coord_nor[-1][i], preds_objectness_nor[-1][i], images_nor.image_sizes[i])
                for i in range(len(batched_inputs))
            ]
            preds_str_from_nor_all = [
                self.create_unknown_instances_from_outputs(preds_class_str_from_nor[-1][i], preds_coord_str_from_nor[-1][i], preds_objectness_str_from_nor[-1][i], images_nor.image_sizes[i])
                for i in range(len(batched_inputs))
            ]
            
            preds_str_all = [
                self.create_unknown_instances_from_outputs(preds_class_str[-1][i], preds_coord_str[-1][i], preds_objectness_str[-1][i], images_str.image_sizes[i])
                for i in range(len(batched_inputs))
            ]
            preds_nor_from_str_all = [
                self.create_unknown_instances_from_outputs(preds_class_nor_from_str[-1][i], preds_coord_nor_from_str[-1][i], preds_objectness_nor_from_str[-1][i], images_str.image_sizes[i])
                for i in range(len(batched_inputs))
            ]                           
                  
            cosine_disagree_nor = [self.get_cosine_disagreement_instances(
                preds_class_nor[-1][i], 
                preds_class_str_from_nor[-1][i],
                preds_nor_all[i],
                preds_str_from_nor_all[i],
                gt_instances_nor_new[i],
                images_nor.image_sizes[i],
                cos_thr=self.sim_threshold,
                obj_thr=self.obj_threshold,
                gt_iou_thr=self.gt_iou_threshold,
            ) for i in range(len(batched_inputs))
            ]
            cosine_disagree_str = [self.get_cosine_disagreement_instances(
                preds_class_str[-1][i], 
                preds_class_nor_from_str[-1][i],
                preds_str_all[i],
                preds_nor_from_str_all[i],
                gt_instances_str_new[i],
                images_str.image_sizes[i],
                cos_thr=self.sim_threshold,
                obj_thr=self.obj_threshold,
                gt_iou_thr=self.gt_iou_threshold,
            ) for i in range(len(batched_inputs))
            ]
            
            #########################################################################
            
            loss_nor_dict = self.criterion(output_nor, targets_nor_new, cosine_disagree_nor)
            loss_str_dict = self.criterion(output_str, targets_str_new, cosine_disagree_str)

            merged_loss_dict = {
                k: loss_nor_dict[k] + loss_str_dict[k] for k in loss_nor_dict.keys()
            }

            weight_dict = self.criterion.weight_dict
            for k in merged_loss_dict.keys():
                if k in weight_dict:
                    merged_loss_dict[k] *= weight_dict[k]

            ###################################################################

            return merged_loss_dict

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                      device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
        box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
        x_start = torch.randn(self.num_proposals, 4, device=self.device)

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        ts = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            ts.append(d_t)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(ts)

    def inference(self, box_cls, box_objectness, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_objectness (Tensor): tensors of shape (batch_size, num_proposals, 1).
                The tensor predicts the objectness for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.sampling_method == 'Random':
            multiple_sample = 1
        else:
            multiple_sample = self.multiple_sample

        if self.disentangled == 0:
            scores = torch.sigmoid(box_cls)
        else:
            scores = torch.softmax(box_cls, dim=-1) * box_objectness
        labels = torch.arange(self.num_classes, device=self.device). \
            unsqueeze(0).repeat(self.num_proposals * multiple_sample, 1).flatten(0, 1)

        for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, box_pred, image_sizes
        )):
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(
                self.num_proposals * multiple_sample, sorted=False)
            labels_per_image = labels[topk_indices]
            box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
            box_pred_per_image = box_pred_per_image[topk_indices]

            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.6)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            # rescale scores to accommodate score threshold
            if self.disentangled == 2:
                scores_per_image[labels_per_image != self.num_classes-1] *= 0.75
                scores_per_image[labels_per_image == self.num_classes-1] *= 2

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        if not self.training:
            images_raw = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
            images_raw = ImageList.from_tensors(images_raw, self.size_divisibility)
            images_nor = None
            images_str = None
            
            images_raw_whwh = list()
            for bi in batched_inputs:
                h_raw, w_raw = bi["image"].shape[-2:]
                images_raw_whwh.append(torch.tensor([w_raw, h_raw, w_raw, h_raw], dtype=torch.float32, device=self.device))
            images_raw_whwh = torch.stack(images_raw_whwh)
            images_nor_whwh = None
            images_str_whwh = None
            
        if self.training:
            images_raw = [self.normalizer(x["image_raw"].to(self.device)) for x in batched_inputs]
            images_raw = ImageList.from_tensors(images_raw, self.size_divisibility)
            images_nor = [self.normalizer(x["image_nor"].to(self.device)) for x in batched_inputs]
            images_nor = ImageList.from_tensors(images_nor, self.size_divisibility)
            images_str = [self.normalizer(x["image_str"].to(self.device)) for x in batched_inputs]
            images_str = ImageList.from_tensors(images_str, self.size_divisibility)

            images_raw_whwh = list()
            images_nor_whwh = list()
            images_str_whwh = list()
            for bi in batched_inputs:
                h_raw, w_raw = bi["image_raw"].shape[-2:]
                h_nor, w_nor = bi["image_nor"].shape[-2:]
                h_str, w_str = bi["image_str"].shape[-2:]
                images_raw_whwh.append(torch.tensor([w_raw, h_raw, w_raw, h_raw], dtype=torch.float32, device=self.device))
                images_nor_whwh.append(torch.tensor([w_nor, h_nor, w_nor, h_nor], dtype=torch.float32, device=self.device))
                images_str_whwh.append(torch.tensor([w_str, h_str, w_str, h_str], dtype=torch.float32, device=self.device))
            images_raw_whwh = torch.stack(images_raw_whwh)
            images_nor_whwh = torch.stack(images_nor_whwh)
            images_str_whwh = torch.stack(images_str_whwh)

        return images_raw, images_nor, images_str, images_raw_whwh, images_nor_whwh, images_str_whwh
