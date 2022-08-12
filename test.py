import imageio
fps=10
writer = imageio.get_writer('video1.avi', fps=fps)

import numpy as np
import cv2
from time import sleep
cap = cv2.VideoCapture('polyp.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out=None
# out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (640,480))
c=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # frame = cv2.flip(frame,0)
        frame_size=(frame.shape[0], frame.shape[1])
        
        if out==None:
            out = cv2.VideoWriter('output1.avi', fourcc, 20.0, frame_size)

        writer.append_data(frame)
        # out.write(frame)

        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # cv2.imwrite(f'pred/i{c}.jpg', frame)
        # frame=np.array(frame)
        # out.write(frame)
        # sleep(.001)
        c+=1
    else:
        break

writer.close()
cap.release()

out.release()

cv2.destroyAllWindows()

























# import cv2
# import torch
# import numpy as np
# vidcap = cv2.VideoCapture('polyp.mp4')
# success, img = vidcap.read()
# #print(img)
# c=0
# p=0
# SIZE =448
# frameSize = (448, 448)
# out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)
# total_time=0
# while success and vidcap.isOpened():
#     success, image = vidcap.read()
#     if success==False:
#         break
    
#     if c>=100 and c<=500:
#         p+=1
#         # transform = transforms.Compose([ 
#         #             transforms.ToTensor(),
#         #             transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                    
#         #             transforms.Resize((SIZE, SIZE)), 
#         #         #    transforms.RandomCrop( SIZE, SIZE)
#         #         ])
#         # image = transform(image)
#         x=image
        
#         out.write(image)
#         print(x.shape, c)
#         x = np.expand_dims(x,0)
#         x=torch.tensor(x)
#         import time
#         start=time.time()
        
#         end=time.time()
#         total_time+=end-start
#         print("Time for one frame: ", end-start)
#     c=c+1














# import cv2 
# path = "pred1/i1.jpg"
# image = cv2.imread(path)
# width, height,_=image.shape
# window_name = 'Image'
# start_point = (5, 5)
# end_point = (5+width, 5+height)
# color = (255, 0, 0)
# thickness = 2
# image = cv2.rectangle(image, start_point, end_point, color, thickness)
# cv2.imshow(window_name, image) 
# cv2.waitKey(0)
















# import matplotlib.pyplot as plt

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

# images = ['video-predict/i1.jpg', 'video-predict/i2.jpg', 'video-predict/i100.jpg']
# plt.axis('off')
# img = None

# for f in images:
#    im = plt.imread(f)
#    if img is None:
#       img = plt.imshow(im)
#       plt.pause(0.5)
#    else:
#       img.set_data(im)
#    plt.pause(0.5)
#    plt.draw()