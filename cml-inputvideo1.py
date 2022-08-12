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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import imageio




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
    print("thresh", threshold)
    bboxes = [box for box in bboxes if box[1] > threshold]
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



##################################################################
fps=10
writer = imageio.get_writer('video1.avi', fps=fps)
###################################################################


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
        

    # for i in range(batch_size):
        
    nms_boxes = non_max_suppression(
            bboxes[0], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        
        # plot_image1(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
    
    individual_frame = plot_image1(x[0].permute(1,2,0).detach().cpu(), nms_boxes, k)

    plt.cla()
        # print(nms_boxes, i+1)
        # print(nms_boxes1, i+1)
        # plot_image1(x[i].permute(1,2,0).detach().cpu(), nms_boxes1, i+1)
fig, ax = plt.subplots(1)



def plot_image1(image, boxes, k):
    
    im = np.array(image)
    im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    height, width, _ = im.shape
    k+=1
    # Create figure and axes
    
    #fig, ax = plt.subplots(1)
    # ax.plot(range(10))

    # fig.patch.set_visible(False)
    # ax.axis('off')


    # Display the image
    ax.imshow(im)
    im=cv2.resize(im, (448,448))
    # cv2.imshow('frame',im)
    # cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
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
        # t = class_labels[int(class_pred)]+"-0.75"
        t=confi*100
        t = int(t)
        t=str(t)+"%"
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=t,
            color="black",
            verticalalignment="top",
            bbox={"color": "white", "pad": 0},
        )

        # print(width, height)
        x1=upper_left_x * width
        y1=upper_left_y * height
        x2=box[2] * width
        y2=box[3] * height

        #im=cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)),(0, 0, 255), 2)
        # Preparing text for the Label
        # label = 'Detected Object'
        # Putting text with Label on the current BGR frame
        # cv2.putText(frame_BGR, label, (x_min - 5, y_min - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    plt.pause(0.1)
    # im=plt.imread(image)





input_thresh=0.8
if len(sys.argv)<5:
    print("usage python3 cml_input.py video.mp4 output.avi frame_number threshold")
else:
    input_thresh=float(sys.argv[4])




if(len(sys.argv)<2):
    print("usage python3 cml_input.py video.mp4")
else:
    input_video=sys.argv[1]
    vidcap = cv2.VideoCapture(input_video)
    success, img = vidcap.read()
    c=0
    p=0
    SIZE =448
    frameSize = (448, 448)
    out = cv2.VideoWriter('d1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)
    total_time=0
    # ax=None
    

    while success and vidcap.isOpened():
        success, image = vidcap.read()
        if success==False:
            break
        
        if c>=0 and c<=1100:
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
            # fig, ax = plt.subplots(1)
            img=plot_couple_examples1(model, x, input_thresh, 0.5, scaled_anchors, c)
            ing=plt.imread(f"pred/i{c}.jpg")


            end=time.time()                                                 
            total_time+=end-start                                                   
            print("Time for one frame: ", end-start)
        c=c+1
        
    
    print(total_time)
writer.close()

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
        # image=Image.open(filename)
        # image=cv2.imread(filename)
        # image_list.append(im)

    # print(frame1[0])
    frame_input=6
    if len(sys.argv)<4:
        print("usage python3 cml_input.py video.mp4 file.avi frame-size  \n By default 6")
    else:
        frame_input=int(sys.argv[3])
    out = cv2.VideoWriter(input_file,cv2.VideoWriter_fourcc(*'DIVX'), frame_input, frameSize)

    for i in frame1:
        out.write(i)
    out.release()







# """
# Course:  Training YOLO v3 for Objects Detection with Custom Data

# Section-1
# Quick Win - Step 2: Simple Object Detection by thresholding with mask
# File: detecting-object.py
# """


# # Detecting Object with chosen Colour Mask
# #
# # Algorithm:
# # Reading RGB image --> Converting to HSV --> Implementing Mask -->
# # --> Finding Contour Points --> Extracting Rectangle Coordinates -->
# # --> Drawing Bounding Box --> Putting Label
# #
# # Result:
# # Window with Detected Object, Bounding Box and Label in Real Time


# # Importing needed library
# import cv2


# # Defining lower bounds and upper bounds of founded Mask
# min_blue, min_green, min_red = 21, 222, 70
# max_blue, max_green, max_red = 176, 255, 255

# # Getting version of OpenCV that is currently used
# # Converting string into the list by dot as separator
# # and getting first number
# v = cv2.__version__.split('.')[0]

# # Defining object for reading video from camera
# camera = cv2.VideoCapture(0)


# # Defining loop for catching frames
# while True:
#     # Capture frame-by-frame from camera
#     _, frame_BGR = camera.read()

#     # Converting current frame to HSV
#     frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

#     # Implementing Mask with founded colours from Track Bars to HSV Image
#     mask = cv2.inRange(frame_HSV,
#                        (min_blue, min_green, min_red),
#                        (max_blue, max_green, max_red))

#     # Showing current frame with implemented Mask
#     # Giving name to the window with Mask
#     # And specifying that window is resizable
#     cv2.namedWindow('Binary frame with Mask', cv2.WINDOW_NORMAL)
#     cv2.imshow('Binary frame with Mask', mask)

#     # Finding Contours
#     # Pay attention!
#     # Different versions of OpenCV returns different number of parameters
#     # when using function cv2.findContours()

#     # In OpenCV version 3 function cv2.findContours() returns three parameters:
#     # modified image, found Contours and hierarchy
#     # All found Contours from current frame are stored in the list
#     # Each individual Contour is a Numpy array of(x, y) coordinates
#     # of the boundary points of the Object
#     # We are interested only in Contours

#     # Checking if OpenCV version 3 is used
#     if v == '3':
#         _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     # In OpenCV version 4 function cv2.findContours() returns two parameters:
#     # found Contours and hierarchy
#     # All found Contours from current frame are stored in the list
#     # Each individual Contour is a Numpy array of(x, y) coordinates
#     # of the boundary points of the Object
#     # We are interested only in Contours

#     # Checking if OpenCV version 4 is used
#     else:
#         contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     # Finding the biggest Contour by sorting from biggest to smallest
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)

#     # Extracting Coordinates of the biggest Contour if any was found
#     if contours:
#         # Getting rectangle coordinates and spatial size from biggest Contour
#         # Function cv2.boundingRect() is used to get an approximate rectangle
#         # around the region of interest in the binary image after Contour was found
#         (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])

#         # Drawing Bounding Box on the current BGR frame
#         cv2.rectangle(frame_BGR, (x_min - 15, y_min - 15),
#                       (x_min + box_width + 15, y_min + box_height + 15),
#                       (0, 255, 0), 3)

#         # Preparing text for the Label
#         label = 'Detected Object'

#         # Putting text with Label on the current BGR frame
#         cv2.putText(frame_BGR, label, (x_min - 5, y_min - 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

#     # Showing current BGR frame with Detected Object
#     # Giving name to the window with Detected Object
#     # And specifying that window is resizable
#     cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
#     cv2.imshow('Detected Object', frame_BGR)

#     # Breaking the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# # Destroying all opened windows
# cv2.destroyAllWindows()


"""
Some comments

With OpenCV function cv2.findContours() we find 
contours of white object from black background.

There are three arguments in cv.findContours() function,
first one is source image, second is contour retrieval mode,
third is contour approximation method.


In OpenCV version 3 three parameters are returned:
modified image, the contours and hierarchy.
Further reading about Contours in OpenCV v3:
https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html


In OpenCV version 4 two parameters are returned:
the contours and hierarchy.
Further reading about Contours in OpenCV v4:
https://docs.opencv.org/4.0.0/d4/d73/tutorial_py_contours_begin.html


Contours is a Python list of all the contours in the image.
Each individual contour is a Numpy array of (x,y) coordinates 
of boundary points of the object.

Contours can be explained simply as a curve joining all the 
continuous points (along the boundary), having same colour or intensity.
"""



























# if len(sys.argv)<3:
#     print("usage python3 cml_input.py video.mp4 file.avi")
# else:
#     input_file=sys.argv[2]
                    
#     frameSize = (448, 448)
#     # out = cv2.VideoWriter('output_1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)

#     frame1=[]
#     # imgpath=[]
#     imgpath1=[]
#     # for filename in glob.glob("video-predict/*.jpg"):
#     #     imgpath.append(filename)
#     #     if p==0:
#     #         break 
#     #     p-=1
#     for file_name in range(1, len(imgpath)):
#         imgpath1.append(f"video-predict/i{file_name}.jpg")

#     # imgpath1.sort()

#     for filename in imgpath1: #assuming gif
#         print(filename)
#         # image = Image.open(filename)
#         image=cv2.imread(filename)
#         # image.show()
#         # cv2.imshow("pp",image)
#         height, width, layers = image.shape
#         frameSize = (width,height)
#         frame1.append(image)
#         # out.write(image)
#         #image=Image.open(filename)
#         #image=cv2.imread(filename)
#         # image_list.append(im)

#     # print(frame1[0])
#     frame_input=6
#     if len(sys.argv)<4:
#         print("usage python3 cml_input.py video.mp4 file.avi frame-size  \n by default 6")
#     else:
#         frame_input=int(sys.argv[3])
#     out = cv2.VideoWriter(input_file,cv2.VideoWriter_fourcc(*'DIVX'), frame_input, frameSize)

#     for i in frame1:
#         # print(i)
#         out.write(i)

#     out.release()

