from tkinter import Image
import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
# from utils import seed_everything




DATASET = 'data'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 16
IMAGE_SIZE = 448
NUM_CLASSES = 1
LEARNING_RATE = [0.0002]
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = [0.3]
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.0
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "weight-pos-neg/wt_max200_lr-0.0002_threshold-0.3.pth.tar"
# CHECKPOINT_FILE = "checkpoint3.pth.tar"
# cvc clinincDB data

# IMG_DIR = DATASET + "/cvc-aug-img/"
# LABEL_DIR = DATASET + "/cvc-aug-labels/"

# IMG_DIR = DATASET + "/etis-aug-img/"
# LABEL_DIR = DATASET + "/etis-aug-labels/"



# IMG_DIR = DATASET + "/aug-img-test/"
# LABEL_DIR = DATASET + "/aug-label-test/"

# IMG_DIR = DATASET + "/neg-combine-img/"
# LABEL_DIR = DATASET + "/neg-combine-labels/"


# IMG_DIR = DATASET + "/neg-pos-combine-img/"
# LABEL_DIR = DATASET + "/neg-pos-combine-labels/"


IMG_DIR = DATASET + "/neg-combine-img/"
LABEL_DIR = DATASET + "/neg-combine-labels/"

# IMG_DIR = DATASET + "/combine-img3/"
# LABEL_DIR = DATASET + "/combine-labels/"
# IMG_DIR = DATASET + "/images/"
# LABEL_DIR = DATASET + "/generate-labels1/"

l11=116/IMAGE_SIZE
l12=90/IMAGE_SIZE
l13=156/IMAGE_SIZE
l14=198/IMAGE_SIZE
l15=373/IMAGE_SIZE
l16=326/IMAGE_SIZE


l21=30/IMAGE_SIZE
l22=61/IMAGE_SIZE
l23=62/IMAGE_SIZE
l24=45/IMAGE_SIZE
l25=59/IMAGE_SIZE
l26=119/IMAGE_SIZE


l31=10/IMAGE_SIZE
l32=13/IMAGE_SIZE
l33=16/IMAGE_SIZE
l34=30/IMAGE_SIZE
l35=33/IMAGE_SIZE
l36=23/IMAGE_SIZE

# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
# ]  # Note these have been rescaled to be between [0, 1]
ANCHORS = [
    [(l11, l12), (l13, l14), (l15, l16)],
    [(l21, l22), (l23, l24), (l25, l26)],
    [(l31, l32), (l33, l34), (l35, l36)],
]  # Note these have been rescaled to be between [0, 1]
# print(l11, l12, l13, l14, l15, l16, l21, l22, l23, l24, l25, l26, l31, l32, l33, l34, l35, l36)

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

PASCAL_CLASSES = [
    "polyp",
    "polyp"
]



