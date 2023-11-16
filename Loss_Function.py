import torch
import torch.nn.functional as F
import math
def compute_iou(box1, box2):
    '''
    :param box1: 预测框BLS
    :param box2: 目标框target_bbox
    :return: iou,即两个框的iou值,形状为[batch_size, 8400],因为每个batch_size预测了8400个框
    :该函数在下方compute_ciou_loss函数中调用
    '''
    intersect_left = torch.max(box1[..., 0], box2[..., 0])
    intersect_top = torch.max(box1[..., 1], box2[..., 1])
    intersect_right = torch.min(box1[..., 2], box2[..., 2])
    intersect_bottom = torch.min(box1[..., 3], box2[..., 3])

    intersect_area = torch.clamp(intersect_right - intersect_left, min=0) * torch.clamp(
        intersect_bottom - intersect_top, min=0)
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = box1_area + box2_area - intersect_area

    iou = intersect_area / (union_area + 1e-6)
    print(iou.shape)
    return iou


def compute_ciou_loss(box1, box2, alpha=1.0, v=0.5, stride=None):
    '''
    :param box1: 预测框BLS
    :param box2: 目标框target_bbox
    :param alpha:
    :param v:
    :param stride: 图片缩放的比例
    :return: CiOU_Loss,形状为[batch_size, 8400],其中为NAN的地方说明此处没有target_bbox,故计算损失时只对有值的求和。
    '''
    if stride: box2 /= stride
    iou = compute_iou(box1, box2)
    box1_center_x = (box1[..., 2] + box1[..., 0]) / 2
    box1_center_y = (box1[..., 3] + box1[..., 1]) / 2
    box2_center_x = (box2[..., 2] + box2[..., 0]) / 2
    box2_center_y = (box2[..., 3] + box2[..., 1]) / 2

    center_distance = torch.pow(box1_center_x - box2_center_x, 2) + torch.pow(box1_center_y - box2_center_y, 2)
    center_distance /= torch.pow(box2_center_x, 2) + torch.pow(box2_center_y, 2)
    iou_center = iou - center_distance

    box1_width = box1[..., 2] - box1[..., 0]
    box1_height = box1[..., 3] - box1[..., 1]
    box2_width = box2[..., 2] - box2[..., 0]
    box2_height = box2[..., 3] - box2[..., 1]
    v_union = (4 / (math.pi ** 2)) * torch.pow(torch.atan(box2_width / box2_height) - torch.atan(box1_width / box1_height),
                                            2)
    print(v_union)
    alpha_angle = v_union / (1 - iou + v_union)

    ciou_loss = 1 - iou + alpha_angle * alpha - center_distance * v

    return ciou_loss

def calculate_pred_regs(pred_bboxes):
    '''
    :param pred_bboxes: 预测框的位置,即网络输出的BLS
    :return: 返回的是预测框的anchor中心点到框边的距离
    :在下面的函数中调用
    '''
    width = pred_bboxes[..., 2] - pred_bboxes[..., 0]
    height = pred_bboxes[..., 3] - pred_bboxes[..., 1]
    # 计算pred_regs（预测的anchors中心点到各边的距离）
    pred_regs = torch.zeros_like(pred_bboxes)
    pred_regs[..., 0] = -width / 2  # 预测框左边距离
    pred_regs[..., 1] = -height / 2  # 预测框上边距离
    pred_regs[..., 2] = width / 2  # 预测框右边距离
    pred_regs[..., 3] = height / 2  # 预测框下边距离

    return pred_regs

def DFL_loss(pred_bboxes, target_bboxes, stride=None):
    '''
    :param pred_bboxes: 预测框,即BLS
    :param target_bboxes: 目标框
    :param stride:
    :return: 回归损失
    :param stride: 图片缩放的比例
    '''
    # 缩放target_bboxes到特征图尺度
    if stride: target_bboxes /= stride
    # 计算pred_regs与target_bboxes之间的回归DFL Loss
    pred_regs = calculate_pred_regs(pred_bboxes)
    reg_loss = F.smooth_l1_loss(pred_regs, target_bboxes)

    # 返回总的损失
    return reg_loss

