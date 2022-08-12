import sys
from model import yv3
from PIL import Image 
import config
import matplotlib.pyplot as plt
import cv2
import torch.optim as optim
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.patches as patches
from PIL import Image
import glob
import cv2



# import keras
# from keras.utils import normalize
# import numpy as np
from utils import (
    intersection_over_union,
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    plot_image,
)

lr=0.0005






model = yv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, lr)
scaled_anchors = (torch.tensor(config.ANCHORS)*torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)
train_loader, test_loader, train_eval_loader = get_loaders(train_csv_path=config.DATASET + "/6exam.csv", test_csv_path=config.DATASET + "/test-aug.csv")
print(train_loader)

# for epoch in range(config.NUM_EPOCHS):
#     plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
#     import sys
#     sys.exit()


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > 0.8]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    #print(torch.tensor(bboxes).shape)
    bboxes_after_nms = []
    c=0
    # if bboxes:
    #     chosen_box = bboxes.pop(0)
    #     bboxes_after_nms.append(chosen_box)
    
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
    return bboxes_after_nms



def plot_couple_examples1(model, loader, thresh, iou_thresh, anchors, k):
    model.eval()
    # x = next(iter(loader))
    x=loader
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
        model.train()

    # for i in range(batch_size):
        
    nms_boxes = non_max_suppression(
            bboxes[0], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        
        # plot_image1(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
    plot_image1(x[0].permute(1,2,0).detach().cpu(), nms_boxes, k)
        # print(nms_boxes, i+1)
        # print(nms_boxes1, i+1)
        # plot_image1(x[i].permute(1,2,0).detach().cpu(), nms_boxes1, i+1)



def plot_image1(image, boxes, k):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = np.array(image)
    im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
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
        #t = class_labels[int(class_pred)]+"-0.75"
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
    # out.write(ax)
    plt.savefig(f"video-predict/i{k}.jpg")
    # plt.imshow("f")
    # plt.savefig(f"pred1/i{k}.jpg")
    # plt.show()








if(len(sys.argv)<2):
    print("usage python3 cml_input.py video.mp4")
else:
    input_video=sys.argv[1]

    vidcap = cv2.VideoCapture(input_video)
    success, img = vidcap.read()
    #print(img)
    c=0
    p=0
    SIZE =448
    frameSize = (448, 448)
    out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)
    total_time=0
    while success and vidcap.isOpened():
        success, image = vidcap.read()
        if success==False:
            break
        
        if c>=0 and c<=5000:
            p+=1
            transform = transforms.Compose([ 
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                       
                       transforms.Resize((SIZE, SIZE)), 
                    #    transforms.RandomCrop( SIZE, SIZE)
                    ])
            image = transform(image)
            x=image
            
       
            print(x.shape, c)
            x = np.expand_dims(x,0)
            x=torch.tensor(x)
            import time
            start=time.time()
            plot_couple_examples1(model, x, 0.3, 0.5, scaled_anchors, c)
            end=time.time()
            total_time+=end-start
            print("Time for one frame: ", end-start)
        c=c+1

    print(total_time)

if len(sys.argv)<3:
    print("usage python3 cml_input.py video.mp4 file.avi")
else:
    input_file=sys.argv[2]
                    
    frameSize = (448, 448)
    # out = cv2.VideoWriter('output_1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)

    frame1=[]
    imgpath=[]
    imgpath1=[]
    for filename in glob.glob("video-predict/*.jpg"):
        imgpath.append(filename)
        if p==0:
            break 
        p-=1
    for file_name in range(1, len(imgpath)):
        imgpath1.append(f"video-predict/i{file_name}.jpg")

    # imgpath1.sort()

    for filename in imgpath1: #assuming gif
        # print(filename)
        # image = Image.open(filename)
        image=cv2.imread(filename)
        # image.show()
        # cv2.imshow("pp",image)
        height, width, layers = image.shape
        frameSize = (width,height)
        frame1.append(image)
        # out.write(image)
        #image=Image.open(filename)
        #image=cv2.imread(filename)
        # image_list.append(im)

    # print(frame1[0])
    frame_input=6
    if len(sys.argv)<4:
        print("usage python3 cml_input.py video.mp4 file.avi frame-size  \n by default 6")
    else:
        frame_input=int(sys.argv[3])
    out = cv2.VideoWriter(input_file,cv2.VideoWriter_fourcc(*'DIVX'), frame_input, frameSize)

    for i in frame1:
        # print(i)
        out.write(i)

    out.release()

