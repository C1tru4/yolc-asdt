# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_detector
from mmdet.core.utils import flip_tensor
from mmdet.models.detectors.single_stage import SingleStageDetector


@DETECTORS.register_module()
class YOLC(SingleStageDetector):
    """Implementation of YOLC

    <https://arxiv.org/abs/2404.06180>.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 distill_cfg=None):
        super(YOLC, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)
        self.distill_cfg = copy.deepcopy(distill_cfg) if distill_cfg is not None else {}
        self.distill_enable = bool(self.distill_cfg.get('enable', False))
        self.teacher = None
        if self.distill_enable:
            teacher_ckpt = self.distill_cfg.get('teacher_ckpt', None)
            if not teacher_ckpt:
                raise ValueError('distill_cfg.teacher_ckpt must be set when enable=True')
            teacher_cfg = dict(
                type='YOLC',
                backbone=copy.deepcopy(backbone),
                neck=copy.deepcopy(neck),
                bbox_head=copy.deepcopy(bbox_head),
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                pretrained=None,
                init_cfg=init_cfg)
            # 注意：teacher_cfg 中已包含 train_cfg 和 test_cfg，不需要再传入
            self.teacher = build_detector(teacher_cfg)
            load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu', strict=False)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        self.kd_weight_global_hm = float(self.distill_cfg.get('kd_weight_global_hm', 1.0))
        self.kd_weight_global_coarse = float(self.distill_cfg.get('kd_weight_global_coarse', 1.0))
        self.kd_weight_local_refine = float(self.distill_cfg.get('kd_weight_local_refine', 1.0))
        self.use_teacher_crop = bool(self.distill_cfg.get('use_teacher_crop', True))
        self.max_patches_per_img = int(self.distill_cfg.get('max_patches_per_img', 4))
        self.min_patch_size = int(self.distill_cfg.get('min_patch_size', 64))
        self.lsm_visualize = bool(self.distill_cfg.get('lsm_visualize', False))
        # lsm_add_border 默认 False，需要通过可视化确认 LSM coords 是否需要加 border
        self.lsm_add_border = bool(self.distill_cfg.get('lsm_add_border', False))

    def train(self, mode=True):
        """保证 student 正常 train，但 teacher 永远 eval（不更新BN统计）。"""
        super().train(mode)
        if getattr(self, 'teacher', None) is not None:
            self.teacher.eval()
        return self

    def _crop_img_tensor(self, img_i, coord, align_size=32):
        """
        Crop image tensor with size alignment for network compatibility.
        
        Args:
            img_i: Image tensor [C, H, W]
            coord: Crop coordinates [x, y, w, h]
            align_size: Alignment size (default 32 for HRNet)
            
        Returns:
            img_patch: Cropped and aligned patch
            offset: Adjusted offset (x1, y1)
        """
        x, y, w, h = [int(round(float(v))) for v in coord]
        _, img_h, img_w = img_i.shape
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)
        
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        # Calculate current patch size
        patch_w = x2 - x1
        patch_h = y2 - y1
        
        # Align patch size to align_size (32) by adjusting coordinates
        # Try to expand first, if exceeding image boundary then shrink
        aligned_w = ((patch_w + align_size - 1) // align_size) * align_size
        aligned_h = ((patch_h + align_size - 1) // align_size) * align_size
        
        # Calculate expansion needed
        expand_w = aligned_w - patch_w
        expand_h = aligned_h - patch_h
        
        # Try to expand symmetrically
        expand_left = expand_w // 2
        expand_right = expand_w - expand_left
        expand_top = expand_h // 2
        expand_bottom = expand_h - expand_top
        
        # Adjust x1, x2, y1, y2
        new_x1 = max(0, x1 - expand_left)
        new_x2 = min(img_w, x2 + expand_right)
        new_y1 = max(0, y1 - expand_top)
        new_y2 = min(img_h, y2 + expand_bottom)
        
        # If still not aligned (hit image boundary), adjust the other side
        actual_w = new_x2 - new_x1
        actual_h = new_y2 - new_y1
        
        if actual_w % align_size != 0:
            target_w = (actual_w // align_size) * align_size
            if target_w < self.min_patch_size:
                target_w = ((actual_w + align_size - 1) // align_size) * align_size
            # Shrink from right first
            if new_x2 - target_w >= 0:
                new_x2 = new_x1 + target_w
            else:
                new_x1 = new_x2 - target_w
        
        if actual_h % align_size != 0:
            target_h = (actual_h // align_size) * align_size
            if target_h < self.min_patch_size:
                target_h = ((actual_h + align_size - 1) // align_size) * align_size
            # Shrink from bottom first
            if new_y2 - target_h >= 0:
                new_y2 = new_y1 + target_h
            else:
                new_y1 = new_y2 - target_h
        
        # Final validation
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_w, new_x2)
        new_y2 = min(img_h, new_y2)
        
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            return None, None
        
        img_patch = img_i[:, new_y1:new_y2, new_x1:new_x2]
        return img_patch, (new_x1, new_y1)

    def _build_patch_meta(self, meta, patch_shape, offset):
        meta_patch = copy.deepcopy(meta)
        patch_h, patch_w = patch_shape
        if isinstance(meta.get('img_shape', None), (list, tuple)) and len(meta['img_shape']) == 3:
            patch_c = meta['img_shape'][2]
        else:
            patch_c = 3
        
        # patch 的当前输入尺寸（经过 crop 后的尺寸）
        meta_patch['img_shape'] = (patch_h, patch_w, patch_c)
        meta_patch['pad_shape'] = (patch_h, patch_w, patch_c)
        meta_patch['batch_input_shape'] = (patch_h, patch_w)
        
        # ori_shape 继承原图的原始尺寸（不要改成 patch 尺寸，避免坐标错位）
        if 'ori_shape' in meta:
            meta_patch['ori_shape'] = meta['ori_shape']
        else:
            meta_patch['ori_shape'] = meta_patch['img_shape']  # 兜底
        
        # 额外记录 patch 自身尺寸（如果需要区分）
        meta_patch['patch_shape'] = (patch_h, patch_w, patch_c)
        
        # patch 继承原图的 scale_factor（patch 和全图在同一缩放体系）
        meta_patch['scale_factor'] = meta.get('scale_factor', 1.0)
        
        # 下面这些是为了避免 patch 被当成"翻转增强图"
        meta_patch['border'] = [0, 0, 0, 0]
        meta_patch['flip'] = False
        meta_patch['flip_direction'] = None
        
        # offset 只存 (x_off, y_off)
        meta_patch['crop_offset'] = (int(offset[0]), int(offset[1]))
        return meta_patch

    def _remap_bboxes_to_full(self, bboxes_patch, crop_offset):
        if bboxes_patch.numel() == 0:
            return bboxes_patch
        x_off, y_off = crop_offset
        bboxes_patch = bboxes_patch.clone()
        bboxes_patch[:, 0] += x_off
        bboxes_patch[:, 2] += x_off
        bboxes_patch[:, 1] += y_off
        bboxes_patch[:, 3] += y_off
        return bboxes_patch

    def _filter_coords(self, coords, img_hw):
        if coords is None or len(coords) == 0:
            return []
        img_h, img_w = img_hw
        filtered = []
        for coord in coords:
            x, y, w, h = [int(round(float(v))) for v in coord]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, img_w)
            y2 = min(y + h, img_h)
            w2 = x2 - x1
            h2 = y2 - y1
            if w2 < self.min_patch_size or h2 < self.min_patch_size:
                continue
            filtered.append([x1, y1, w2, h2])
        if self.max_patches_per_img > 0:
            filtered = filtered[:self.max_patches_per_img]
        return filtered

    def _get_patch_coords(self, center_heatmap_preds, img_metas, img_hw):
        coords_list = []
        for img_idx in range(len(img_metas)):
            heatmap_i = [center_heatmap_preds[0][img_idx:img_idx + 1]]
            metas_i = [img_metas[img_idx]]
            coords = self.bbox_head.LSM(heatmap_i, metas_i, visualize=self.lsm_visualize)
            
            # LSM 返回的坐标可能已经减去了 border（取决于 LSM 实现）
            # 如果启用 lsm_add_border，则加回 border 才能在 img tensor 上正确裁剪
            # 注意：border 格式在不同 mmdet 版本/pipeline 中可能不一致
            #       可能是 [top, bottom, left, right] 或 [left, top, right, bottom]
            #       建议先可视化确认 coords 是否正确，再决定是否启用此修正
            if self.lsm_add_border and coords is not None and len(coords) > 0:
                border = metas_i[0].get('border', [0, 0, 0, 0])
                coords_adjusted = []
                for coord in coords:
                    x, y, w, h = coord
                    # 假设 border 格式是 [top, bottom, left, right]
                    x_adjusted = x + border[2]  # 加上 left border
                    y_adjusted = y + border[0]  # 加上 top border
                    coords_adjusted.append([x_adjusted, y_adjusted, w, h])
                coords = coords_adjusted
            coords_list.append(self._filter_coords(coords, img_hw[img_idx]))
        return coords_list

    def _run_detector_once(self, model, img_patch, meta_patch):
        feat = model.extract_feat(img_patch)
        return model.bbox_head(feat), meta_patch

    def simple_test(self, img, img_metas, rescale=False, crop=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        if crop:
            maxcontours, results_list = self.bbox_head.simple_test(
                feat, img_metas, rescale=rescale, crop=crop)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return maxcontours, bbox_results

        # 使用原版 simple_test 获取全图结果（保持原 YOLC 推理行为一致）
        # rescale=False：全图结果保持在当前输入图尺度（img_shape），不还原到原图尺度
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=False, crop=False)
        
        # 获取 heatmap 用于生成 patch coords
        outs = self.bbox_head(feat)
        coords_source = outs[0]

        img_hw = [(img_i.shape[-2], img_i.shape[-1]) for img_i in img]
        coords_list = self._get_patch_coords(coords_source, img_metas, img_hw)

        merged_results = []
        for img_idx, (det_bboxes, det_labels) in enumerate(results_list):
            patch_coords = coords_list[img_idx]
            if patch_coords:
                for coord in patch_coords:
                    img_patch, offset = self._crop_img_tensor(img[img_idx], coord)
                    if img_patch is None:
                        continue
                    meta_patch = self._build_patch_meta(img_metas[img_idx], img_patch.shape[-2:], offset)
                    outs_patch, meta_patch = self._run_detector_once(self, img_patch.unsqueeze(0), [meta_patch])
                    patch_det = self.bbox_head.get_bboxes(
                        *outs_patch, meta_patch, rescale=False, with_nms=False)[0]
                    patch_bboxes, patch_labels = patch_det
                    # 确认 bbox 格式并跳过无效检测
                    if patch_bboxes.numel() == 0:
                        continue
                    if patch_bboxes.shape[-1] != 5:
                        # 如果格式不符合预期，跳过该 patch（调试时可改为 assert）
                        continue
                    patch_bboxes = self._remap_bboxes_to_full(patch_bboxes, meta_patch[0]['crop_offset'])
                    det_bboxes = torch.cat([det_bboxes, patch_bboxes], dim=0)
                    det_labels = torch.cat([det_labels, patch_labels], dim=0)

            # 融合后重新做 NMS（全图框 + patch 框合并后统一做 NMS）
            nms_cfg = self.bbox_head.test_cfg.get('nms_cfg', None)
            if nms_cfg is not None and det_bboxes.numel() > 0:
                det_bboxes, det_labels = self.bbox_head._bboxes_nms(
                    det_bboxes, det_labels, self.bbox_head.test_cfg)

            # rescale：将当前输入图尺度的框还原到原图尺度
            # 前提：全图和 patch 都用 rescale=False，框都在 img_shape 尺度
            # scale_factor 是从原图到当前输入图的缩放比例
            if rescale and det_bboxes.numel() > 0:
                scale_factor = det_bboxes.new_tensor(img_metas[img_idx].get('scale_factor', 1.0))
                det_bboxes[:, :4] = det_bboxes[:, :4] / scale_factor

            merged_results.append((det_bboxes, det_labels))

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in merged_results
        ]
        return bbox_results

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        feat_s = self.extract_feat(img)
        outs_s = self.bbox_head(feat_s)
        losses = self.bbox_head.loss(
            *outs_s, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=gt_bboxes_ignore)

        if not self.distill_enable or self.teacher is None:
            # 添加统计项保持日志格式统一（即使 KD 未启用）
            losses['kd_patch_cnt'] = img.new_tensor(0.0)
            losses['kd_enable'] = img.new_tensor(0.0)
            return losses

        with torch.no_grad():
            feat_t = self.teacher.extract_feat(img)
            outs_t = self.teacher.bbox_head(feat_t)

        hm_s = outs_s[0][0]
        hm_t = outs_t[0][0].detach()
        kd_hm = F.mse_loss(torch.sigmoid(hm_s), torch.sigmoid(hm_t), reduction='mean')
        losses['loss_kd_global_hm'] = kd_hm * self.kd_weight_global_hm

        bbox_c_s = self.bbox_head.decode_xywh_to_bbox(outs_s[1][0], img_metas)
        bbox_c_t = self.teacher.bbox_head.decode_xywh_to_bbox(outs_t[1][0], img_metas).detach()
        kd_coarse = F.mse_loss(bbox_c_s, bbox_c_t, reduction='mean')
        losses['loss_kd_global_coarse'] = kd_coarse * self.kd_weight_global_coarse

        if self.use_teacher_crop:
            coords_source = outs_t[0]
        else:
            coords_source = outs_s[0]

        img_hw = [(img_i.shape[-2], img_i.shape[-1]) for img_i in img]
        coords_list = self._get_patch_coords(coords_source, img_metas, img_hw)

        kd_local_sum = img.new_tensor(0.0)
        patch_cnt = 0
        for img_idx, patch_coords in enumerate(coords_list):
            for coord in patch_coords:
                img_patch, offset = self._crop_img_tensor(img[img_idx], coord)
                if img_patch is None:
                    continue
                meta_patch = self._build_patch_meta(img_metas[img_idx], img_patch.shape[-2:], offset)

                feat_s_patch = self.extract_feat(img_patch.unsqueeze(0))
                outs_s_patch = self.bbox_head(feat_s_patch)

                with torch.no_grad():
                    feat_t_patch = self.teacher.extract_feat(img_patch.unsqueeze(0))
                    outs_t_patch = self.teacher.bbox_head(feat_t_patch)

                bbox_r_s = self.bbox_head.decode_xywh_to_bbox(outs_s_patch[2][0], [meta_patch])
                bbox_r_t = self.teacher.bbox_head.decode_xywh_to_bbox(outs_t_patch[2][0], [meta_patch]).detach()
                kd_local_sum += F.mse_loss(bbox_r_s, bbox_r_t, reduction='mean')
                patch_cnt += 1

        if patch_cnt > 0:
            kd_local = kd_local_sum / patch_cnt
        else:
            kd_local = img.new_tensor(0.0)
        losses['loss_kd_local_refine'] = kd_local * self.kd_weight_local_refine
        
        # 添加 KD 监控统计项（不参与梯度，仅用于日志观察）
        losses['kd_patch_cnt'] = img.new_tensor(float(patch_cnt))
        losses['kd_enable'] = img.new_tensor(1.0 if self.distill_enable else 0.0)
        
        return losses


    def merge_aug_results(self, aug_results, with_nms):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            with_nms (bool): If True, do nms before return boxes.

        Returns:
            tuple: (out_bboxes, out_labels)
        """
        recovered_bboxes, aug_labels = [], []
        for single_result in aug_results:
            recovered_bboxes.append(single_result[0][0])
            aug_labels.append(single_result[0][1])

        bboxes = torch.cat(recovered_bboxes, dim=0).contiguous()
        labels = torch.cat(aug_labels).contiguous()
        if with_nms:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=True):
        """Augment testing of CenterNet. Aug test must have flipped image pair,
        and unlike CornerNet, it will perform an averaging operation on the
        feature map instead of detecting bbox.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            flip_direction = img_metas[flip_ind][0]['flip_direction']
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            center_heatmap_preds, xywh_preds_coarse, xywh_preds_refine = self.bbox_head(x)
            assert len(center_heatmap_preds) == len(xywh_preds_coarse) == len(xywh_preds_refine)  == 1

            # Feature map averaging
            center_heatmap_preds[0] = (
                center_heatmap_preds[0][0:1] +
                flip_tensor(center_heatmap_preds[0][1:2], flip_direction)) / 2
            xywh_preds_refine[0] = xywh_preds_refine[0][0:1]

            bbox_list = self.bbox_head.get_bboxes(
                center_heatmap_preds,
                xywh_preds_coarse,
                xywh_preds_refine,
                img_metas[ind],
                rescale=rescale,
                with_nms=False)
            aug_results.append(bbox_list)

        nms_cfg = self.bbox_head.test_cfg.get('nms_cfg', None)
        if nms_cfg is None:
            with_nms = False
        else:
            with_nms = True
        bbox_list = [self.merge_aug_results(aug_results, with_nms)]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
