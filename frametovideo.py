path ="pred1"
from PIL import Image
import glob
image_list = []
k=0
import cv2

frameSize = (448, 448)
# out = cv2.VideoWriter('output_1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)

frame1=[]
imgpath=[]
imgpath1=[]
for filename in glob.glob("video-predict/*.jpg"):
    imgpath.append(filename)
for file_name in range(1, len(imgpath)):
    imgpath1.append(f"video-predict/i{file_name}.jpg")

# imgpath1.sort()

for filename in imgpath1: #assuming gif
    print(filename)
    # image = Image.open(filename)
    image=cv2.imread(filename)
    # image.show()
    # cv2.imshow("pp",image)
    height, width, layers = image.shape
    frameSize = (width,height)
    frame1.append(image)
    
out = cv2.VideoWriter('out_2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

for i in frame1:
    out.write(i)

out.release()



