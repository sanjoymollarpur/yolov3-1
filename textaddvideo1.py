import cv2
vidcap = cv2.VideoCapture('polyp.mp4')
success, img = vidcap.read()
c=0
while success and vidcap.isOpened():
    success, image = vidcap.read()
    
    
    if c>=100 and c<=130:
       font = cv2.FONT_HERSHEY_SIMPLEX
       f=f"text->{c}"
       cv2.putText(image,
				f,
				(50, 50),
				font, .5,
				(255, 0, 255),
				1,
				cv2.LINE_4)
       cv2.imwrite(f"metric/{c}.jpg",image)
       # cv2.imshow('video', image)
    if c>=120 and c<=150:
       font = cv2.FONT_HERSHEY_SIMPLEX
       f=f"text-2->{c}"
       cv2.putText(image,
				f,
				(60, 60),
				font, .5,
				(255, 0, 255),
				1,
				cv2.LINE_4)
       cv2.imwrite(f"metric/{c}.jpg",image)
        
    c=c+1










