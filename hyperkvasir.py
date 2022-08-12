from model import yv3
from PIL import Image 
import config
import matplotlib.pyplot as plt
import cv2
import torch.optim as optim
import torch
import numpy as np
import torchvision.transforms as transforms


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
    non_max_suppression
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


def plot_couple_examples1(model, loader, thresh, iou_thresh, anchors,k):
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
    plot_image(x[0].permute(1,2,0).detach().cpu(), nms_boxes, k)
        # print(nms_boxes, i+1)
        # print(nms_boxes1, i+1)
        # plot_image1(x[i].permute(1,2,0).detach().cpu(), nms_boxes1, i+1)




SIZE =448
path ="/home/mimyk/Desktop/AI_Work/yolo/yolov3-correct-aug/manual1-yolov3/hyper-kvasir/lower-gi-tract/pathological-findings/ulcerative-colitis-grade-0-1/polyps"
from PIL import Image
import glob
image_list = []
k=0
for filename in glob.glob("hyper-kvasir/lower-gi-tract/pathological-findings/ulcerative-colitis-grade-0-1/*.jpg"): #assuming gif
# for filename in glob.glob("hyper-kvasir/lower-gi-tract/pathological-findings/polyps/*.jpg"):
    print(filename)
    #image=Image.open(filename)
    image=cv2.imread(filename)
    # image_list.append(im)

    transform = transforms.Compose([ 
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                       transforms.Resize((SIZE, SIZE)), 
                    ])
    image = transform(image)
    x=image
    x = np.expand_dims(x,0)
    print(x.shape)
    x=torch.tensor(x)
    
    plot_couple_examples1(model, x, 0.6, 0.5, scaled_anchors,k)
    k+=1










# vidcap = cv2.VideoCapture('polyp.mp4')
# success, img = vidcap.read()
# #print(img)
# c=0
# p=0
# scale=1.1
# SIZE =448
# frameSize = (448, 448)
# out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)

# while success and vidcap.isOpened():
#     success, image = vidcap.read()
    
    
#     if c>=1000 and c<=1010:
#         #cv2.imwrite("/home/prithvi/Desktop/sanjoy/U-Net/code1/unet2/images/%d.png" % p, image)
#         transform = transforms.Compose([ 
#                        transforms.ToTensor(),
#                        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
#                        transforms.Resize((SIZE, SIZE)), 
#                     ])
#         image = transform(image)
#         x=image
#         # x=np.array(x)
#         #x=x.resize(3,SIZE, SIZE)
#         print(x.shape)
#         # x=torch.tensor(x)
#         # x=Image.fromarray(x)
#         # x=np.asarray(x)
#         # x=x.resize((SIZE, SIZE))
#         x = np.expand_dims(x,0)
#         print(x.shape)
#         x=torch.tensor(x)
#         plot_couple_examples1(model, x, 0.6, 0.5, scaled_anchors)






#         #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # test_img_other = image
#         # test_img_other = Image.fromarray(test_img_other)
#         # # test_img_other=np.asarray(test_img_other)
#         # print(test_img_other.mode)
#         # print(test_img_other.size)
       
#         # test_img_other = test_img_other.resize((SIZE, SIZE))
#         # #test_img_other = test_img_other.astype('float32')
#         # # normalize to the range 0-1
#         # #test_img_other /= 255.0
#         # x=test_img_other
#         # x=np.asarray(x)
#         # test_img_other=np.asarray(test_img_other)
        
#         # #out.write(x)
#         # print(x.shape)
#         # #x = x.to("cuda")
#         # test_img_other_norm = np.expand_dims(test_img_other,0)
#         # test_img_other_norm = test_img_other_norm.transpose((0,3,1,2))
#         # #plot_couple_examples1(model, test_img_other_norm, 0.6, 0.5, scaled_anchors)
#         # print(test_img_other_norm.shape)
#         # test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
#         # test_img_other_input=np.expand_dims(test_img_other_norm, 0)
#         # prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.5).astype(np.uint8)


#         # plt.figure(figsize=(16, 8))
#         # plt.subplot(234)
#         # plt.title('Video Frame')
#         # plt.imshow(x)
#         #plt.show()

#         # # plt.subplot(235)
#         # # plt.title('Prediction mask')
#         # # plt.imshow(prediction_other, cmap='gray')
#         #plt.show()
#         p+=1
#     c=c+1





