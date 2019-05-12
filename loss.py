#

import torch
import torch.nn as nn
import utils

class YoloLoss(nn.modules.loss._Loss):
    """ 
    Computes yolo loss from darknet network output and target annotation.
    """
    def __init__(self, noobject_scale=1.0, object_scale=1.0, ignore_thres=0.5):
        super().__init__()
        
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.ignore_thres = ignore_thres
        # criterion
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()


    def forward(self, output, dim_info, target, scaled_anchors, stride):

        # print(output.shape)
        # torch.Size([8, 3, 13, 13, 85])
        # print(target.shape)
        # torch.Size([6])

        pred_boxes = output[..., :4] / stride
        pred_conf = output[..., 4]
        pred_cls = output[..., 5:]
        x, y, w, h = dim_info[..., 0], dim_info[..., 1], dim_info[..., 2], dim_info[..., 3]

        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = utils.build_targets(
            pred_boxes = pred_boxes,
            pred_cls = pred_cls,
            target = target,
            anchors = scaled_anchors,
            ignore_thres = self.ignore_thres,
        )

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        
        loss_conf = self.object_scale * loss_conf_obj + self.noobject_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return total_loss
