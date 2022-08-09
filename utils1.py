import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import cv2

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """
    
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    c=0
    chosen_box=[]
    if len(bboxes)>0:
        chosen_box = bboxes.pop(0)
        bboxes_after_nms.append(chosen_box)
    

    for box in bboxes:
        cb=0
        for box11 in bboxes_after_nms:
            ious= intersection_over_union(torch.tensor(torch.tensor(box11[2:])), torch.tensor(box[2:]),box_format=box_format)
            if ious<=0:
                cb+=1
        if cb==len(bboxes_after_nms):
            bboxes_after_nms.append(box)
    # for box in bboxes:
    #     ious= intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]),box_format=box_format)
    #     if(ious<=iou_threshold):
    #         bboxes_after_nms.append(box)
    return bboxes_after_nms

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=1
):
    average_precisions = []
    iou_score=0
    epsilon = 1e-6

    # for c in range(num_classes):
    detections = pred_boxes
    ground_truths = true_boxes

    print("bx", len(pred_boxes))
    if pred_boxes:
        print(pred_boxes[0])
    print(len(true_boxes))
    # for detection in pred_boxes:
    #     if detection[1] == 0:
    #         detections.append(detection)

    # for true_box in true_boxes:
    #     if true_box[1] == 0:
    #         ground_truths.append(true_box)

    amount_bboxes = Counter([gt[0] for gt in ground_truths])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    # sort by box probabilities which is index 2
    detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)
    
    # If none exists for this class then we can safely skip
    # if total_true_bboxes == 0:
    #     continue
    c1=0
    for detection_idx, detection in enumerate(detections):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [
            bbox for bbox in ground_truths if bbox[0] == detection[0]
        ]


        num_gts = len(ground_truth_img)
        best_iou = 0
        c1+=1
        for idx, gt in enumerate(ground_truth_img):
            iou = intersection_over_union(
                torch.tensor(detection[3:]),
                torch.tensor(gt[3:]),
                box_format=box_format,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        iou_score+=best_iou
        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # if IOU is lower then the detection is a false positive
        else:
            FP[detection_idx] = 1
    #print("TP ", TP)
    TP_cumsum = torch.cumsum(TP, dim=0)
    #print("TP_cumsum ", TP_cumsum)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    # print("p ", torch.sum(precisions))
    # print("r ", torch.sum(recalls))
    # p= sum(precisions)/len(precisions)
    # r= sum(recalls)/len(recalls)
    s1=TP_cumsum/(FP_cumsum+total_true_bboxes+epsilon)
    # s1 = torch.cat((torch.tensor([1]), s1))
    s1= sum(s1)/(1e-4+len(s1))
    # s1=0
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    if len(precisions)>0:
        print("pre ",(precisions[len(precisions)-1]).item())
        print("recall ",(recalls[len(recalls)-1]).item())
    
    # print(precisions)
    # print(recalls)
    p= sum(precisions)/len(precisions)
    r= sum(recalls)/len(recalls)
    # torch.trapz for numerical integration
    f1=2*(r*p)/(r+p+1e-6)
    fp=torch.sum(FP)
    tp=torch.sum(TP)

    p11=tp/(tp+fp+1e-6)
    r11=tp/(total_true_bboxes+1e-6)
    s11=tp/(fp+total_true_bboxes+1e-6)
    f11=2*(p11*r11)/(p11+r11+1e-6)
    print("f11 s11",f11, s11)

    print("fp tp", fp, tp)
    print("f1 ", f1)
    average_precisions.append(torch.trapz(precisions, recalls))
    return sum(average_precisions) / len(average_precisions), s11, f11


def plot_image(image, boxes, k):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    
    
    
    #im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    
    
    
    
    height, width, _ = im.shape
    k+=1
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    for box in boxes:
        assert len(box) == 6 #"box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        confi=box[1]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        # Add the patch to the Axes
        # t = confi
        t=confi*100
        t = int(t)
        t=str(t)+"%"
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=t,
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    plt.savefig(f"pred2/n_0.3-0.0002_i{k}.jpg")
    plt.show()


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)
        with torch.no_grad():
            predictions = model(x)
        
        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        
        for i in range(3):
            S = predictions[i].shape[2]
            #print(predictions[i].shape)
            #print("get evlaute boxes S", S)
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            #print("get evlaute boxes anchor", anchor)
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
    
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )
        
        for idx in range(batch_size):
            
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
        
    # model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to                
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    # print("num_anchors ",num_anchors)
    box_predictions = predictions[..., 1:5]
    #print("box prediction ",box_predictions.shape)
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        # print("anchors ",anchors.shape)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]
   
    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    # print("cell indices", cell_indices.shape)
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            ) 
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()
    loss=[]
    l1=f"{(correct_class/(tot_class_preds+1e-16))*100:2f}"
    l2= f"{(correct_noobj/(tot_noobj+1e-16))*100:2f}"
    l3 = f"{(correct_obj/(tot_obj+1e-16))*100:2f}"

    loss.append(float(l1))
    # loss.append(float(l2))
    # loss.append(float(l3))
    return float(l3), float(l2)


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path):
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
    
    print(y[0].shape)

    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        bboxes1 = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = y[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                y[i], anchor, S=S, is_preds=False
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes1[idx] += box
        
        model.train()

    for i in range(batch_size):
        
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        nms_boxes1 = non_max_suppression(
            bboxes1[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        #plot_image1(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, i+1)
        # print(nms_boxes, i+1)
        # print(nms_boxes1, i+1)
        # plot_image1(x[i].permute(1,2,0).detach().cpu(), nms_boxes1, i+1)

def plot_image1(image, boxes, k):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    # im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    height, width, _ = im.shape
    k+=1
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6 #"box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        confi = box[1]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )

        # Add the patch to the Axes
        # t = confi
        t=confi*100
        t = int(t)
        t=str(t)+"%"
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=t,
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
        
    plt.savefig(f"pred/n_0.3-0.0002_i{k}.jpg")
    plt.show()





# def plot_image1(image, boxes):
#     """Plots predicted bounding boxes on the image"""
#     cmap = plt.get_cmap("tab20b")
#     class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
#     colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
#     im = np.array(image)
#     height, width, _ = im.shape

#     # Create figure and axes
#     fig, ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(im)

#     # box[0] is x midpoint, box[2] is width
#     # box[1] is y midpoint, box[3] is height

#     # Create a Rectangle patch
#     for box in boxes:
#         assert len(box) == 6 #"box should contain class pred, x, y, width, height"
#         class_pred = box[0]
#         box = box[1:]
#         upper_left_x = box[0] - box[2] / 2
#         upper_left_y = box[1] - box[3] / 2
#         rect = patches.Rectangle(
#             (upper_left_x * width, upper_left_y * height),
#             box[2] * width,
#             box[3] * height,
#             linewidth=2,
#             edgecolor="red",
#             facecolor="none",
#         )
#         # Add the patch to the Axes
#         ax.add_patch(rect)
#         plt.text(
#             upper_left_x * width,
#             upper_left_y * height,
#             s=class_labels[int(class_pred)],
#             color="white",
#             verticalalignment="top",
#             bbox={"color": colors[int(class_pred)], "pad": 0},
#         )
#     plt.show()




def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False







