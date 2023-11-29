import cv2
import numpy as np

def cam(ip):
    user='admin'
    passwd='zx784512'
    ch='1'
    cam=cv2.VideoCapture('rtsp://'+
                         user +
                         ':'+
                         passwd +
                         '@'+
                         ip+
                         '/cam/realmonitor?channel='+
                         ch +
                         '&subtype=0')

    return cam

ip='192.168.100.69'
#ip='192.168.100.28'
src=cam(ip)

while 1:
    r,f=src.read()
    if not(r):
        continue
    #img=cv2.resize(f,(800,600))
    img=f.copy()
    x,y,c=img.shape
    pts1 = np.float32([[100,100], [500,100], [100, 400], [500,400]])
    pts2 = np.float32([[100,100], [450,100], [300,400], [500,400]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (x,y))
    
    cv2.imshow('image',dst)
    #cv2.moveWindow('image',10,10)   
    k = cv2.waitKey(1)
    if k == ord('s'):
        break
    
cv2.destroyAllWindows()
