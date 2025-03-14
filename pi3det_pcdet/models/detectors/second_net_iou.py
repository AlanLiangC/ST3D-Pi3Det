import torch
from .detector3d_template import Detector3DTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class SECONDNetIoU(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def det_source_stage_mode(self):
        for param in self.parameters():
            param.requires_grad = True

        if hasattr(self.dense_head, 'alignment_model'):
            for param in self.dense_head.alignment_model.parameters():
                param.requires_grad = False

    def det_target_stage_mode(self):

        if hasattr(self.dense_head, 'alignment_model'):
            for param in self.dense_head.alignment_model.parameters():
                param.requires_grad = True

    def dist_stage_mode(self):
        for param in self.parameters():
            param.requires_grad = False

        if hasattr(self.dense_head, 'alignment_model'):
            for param in self.dense_head.alignment_model.parameters():
                param.requires_grad = True

            for param in self.roi_head.alignment_model.parameters():
                param.requires_grad = True

    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            weights = batch_dict.get('SEP_LOSS_WEIGHTS', None)
            loss, tb_dict, disp_dict = self.get_training_loss(weights)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing_multicriterion(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, weights=None):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss(weights)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        iou_weight = 1.0
        if weights is not None:
            iou_weight = weights[-1]

        loss = loss_rpn + iou_weight * loss_rcnn
        return loss, tb_dict, disp_dict
