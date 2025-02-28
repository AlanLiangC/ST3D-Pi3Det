import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .anchor_head_template import AnchorHeadTemplate

class AlignmentModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(AlignmentModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                        if in_channels != out_channels else None
        self.fc1 = nn.Linear(out_channels, out_channels // reduction)
        self.fc2 = nn.Linear(out_channels // reduction, out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out1 = self.relu(self.branch1(x))
        out2 = self.relu(self.branch3(x))
        out3 = self.relu(self.branch5(x))
        out = out1 + out2 + out3
        w = F.adaptive_avg_pool2d(out, output_size=1)
        w = w.view(w.size(0), -1)                     
        w = self.relu(self.fc1(w))                    
        w = torch.sigmoid(self.fc2(w))                
        w = w.view(w.size(0), w.size(1), 1, 1)        
        out = out * w                                
        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x
        out = out + residual
        return self.relu(out)

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        # ALIGNMENT MODULE
        if self.model_cfg.get('ALIGNMENT', False):
            self.alignment_model = AlignmentModule(input_channels,input_channels,8)

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        if hasattr(self, 'alignment_model') and data_dict['mode'] == 'alignment':
            spatial_features_2d = self.alignment_model(spatial_features_2d)
        
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
