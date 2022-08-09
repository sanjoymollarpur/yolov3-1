# from model import yv3
# from PIL import Image 
# import config
# import matplotlib.pyplot as plt
# import cv2
# import torch.optim as optim
# import torch


# # import keras
# # from keras.utils import normalize
# # import numpy as np
# # from utils import (
# #     intersection_over_union,
# #     mean_average_precision,
# #     cells_to_bboxes,
# #     get_evaluation_bboxes,
# #     save_checkpoint,
# #     load_checkpoint,
# #     check_class_accuracy,
# #     get_loaders,
# #     plot_couple_examples
# # )

# lr=0.0002






# # model = yv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
# # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
# # load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, lr)


# # scaled_anchors = (torch.tensor(config.ANCHORS)*torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)

# # train_loader, test_loader, train_eval_loader = get_loaders(train_csv_path=config.DATASET + "/6exam.csv", test_csv_path=config.DATASET + "/test-aug.csv")
# # print(train_loader)

# # for epoch in range(config.NUM_EPOCHS):
# #     plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)


# vidcap = cv2.VideoCapture('polyp.mp4')
# success, img = vidcap.read()
# #print(img)
# c=0
# p=0
# SIZE =448
# frameSize = (448, 448)
# out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)

# while success and vidcap.isOpened():
#     success, image = vidcap.read()
    
    
#     if c>=3000 and c<=3001:
#         #cv2.imwrite("/home/prithvi/Desktop/sanjoy/U-Net/code1/unet2/images/%d.png" % p, image)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         test_img_other = gray
#         test_img_other = Image.fromarray(test_img_other)
#         test_img_other = test_img_other.resize((SIZE, SIZE))
#         x=test_img_other
#         x=np.array(x)
#         #out.write(x)

#         x = x.to("cuda")
#         test_img_other_norm = np.expand_dims(np.array(test_img_other),2)
#         # test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
#         # test_img_other_input=np.expand_dims(test_img_other_norm, 0)
#         # prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.5).astype(np.uint8)

        
#         plt.figure(figsize=(16, 8))
#         plt.subplot(234)
#         plt.title('Video Frame')
#         plt.imshow(x)

#         # # plt.subplot(235)
#         # # plt.title('Prediction mask')
#         # #plt.imshow(prediction_other, cmap='gray')
#         plt.show()
#         p+=1
#     c=c+1




#from model import yv3
from PIL import Image 
#import config
import matplotlib.pyplot as plt
import cv2
import torch.optim as optim
import torch
import numpy as np


# import keras
# from keras.utils import normalize
# import numpy as np
# from utils import (
#     intersection_over_union,
#     mean_average_precision,
#     cells_to_bboxes,
#     get_evaluation_bboxes,
#     save_checkpoint,
#     load_checkpoint,
#     check_class_accuracy,
#     get_loaders,
#     plot_couple_examples
# )

lr=0.0002






# model = yv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
# load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, lr)


# scaled_anchors = (torch.tensor(config.ANCHORS)*torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)

# train_loader, test_loader, train_eval_loader = get_loaders(train_csv_path=config.DATASET + "/6exam.csv", test_csv_path=config.DATASET + "/test-aug.csv")
# print(train_loader)

# for epoch in range(config.NUM_EPOCHS):
#     plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)


vidcap = cv2.VideoCapture('polyp.mp4')
success, img = vidcap.read()
#print(img)
c=0
p=0
SIZE =448
frameSize = (448, 448)
out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)

while success and vidcap.isOpened():
    success, image = vidcap.read()
    
    
    if c>=3000 and c<=3001:
        #cv2.imwrite("/home/prithvi/Desktop/sanjoy/U-Net/code1/unet2/images/%d.png" % p, image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_img_other = gray
        test_img_other = Image.fromarray(test_img_other)
        test_img_other = test_img_other.resize((SIZE, SIZE))
        x=test_img_other
        x=np.asarray(x)
        #out.write(x)
        print(x.shape)
        #x = x.to("cuda")
        test_img_other_norm = np.expand_dims(np.array(test_img_other),0)
        print(test_img_other_norm.shape)
        # test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
        # test_img_other_input=np.expand_dims(test_img_other_norm, 0)
        # prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.5).astype(np.uint8)

        
        plt.figure(figsize=(16, 8))
        plt.subplot(234)
        plt.title('Video Frame')
        plt.imshow(x)

        # # plt.subplot(235)
        # # plt.title('Prediction mask')
        # # plt.imshow(prediction_other, cmap='gray')
        plt.show()
        p+=1
    c=c+1






