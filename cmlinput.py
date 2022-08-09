from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch.optim as optim
import torch
import numpy as np
lr=0.0002
import sys

if(len(sys.argv)<2):
    print("usage python3 cml_input.py video.mp4")
else:
    input_video=sys.argv[1]

    vidcap = cv2.VideoCapture(input_video)
    success, img = vidcap.read()
    # print(img)
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
            if len(sys.argv)<3:
                print("usage python3 cml_input.py video.mp4 file.jpg")
            else:
                input_file=sys.argv[2]
                plt.savefig(f"pred/{c}{input_file}")
            # # plt.subplot(235)
            # # plt.title('Prediction mask')
            # # plt.imshow(prediction_other, cmap='gray')
            plt.show()
            p+=1
        c=c+1




