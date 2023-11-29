from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2
import time

yolov7 = YOLOv7()
yolov7.set(ocr_classes=['truck','bus','car'])
yolov7.load('coco.weights', classes='coco.yaml', device='cpu') 

def ocrResult(image):
    global detections
    detections = yolov7.detect(image)
    detected_image = draw(image, detections)
    cv2.imshow('detect',detected_image)
    r,d='',''
    for n in detections:
        if (n['confidence']*100>50):
            if 'text' in n.keys():
                plate(n['confidence'],n['text'])
                '''
                print('%s %s %s\n'%(n['class'],
                                    n['confidence'],
                                    n['text']))
                '''
def plate(x,t):
    t=t.split(';')
    for n in t:
        q=n.replace('-','')
        q=q.replace(' ','')
        if len(q)==6:
            print(x,q)

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
fa=None
while 1:
    r,frame=src.read()
    if frame is None:
        continue
    f = frame[350:650,400:1200]
    ff=f.copy()
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if fa is None:
        fa=gray
        continue
    delta=cv2.absdiff(fa,gray)
    thres=cv2.threshold(delta,30,255,cv2.THRESH_BINARY)[1]
    thres=cv2.dilate(thres,None,iterations=2)
    cn,_=cv2.findContours(thres.copy(),
                          cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_SIMPLE)
    for c in cn:
        if not(1000<cv2.contourArea(c)<10000):
            continue
        ocrResult(ff)
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,0),2)
        fa=gray
        
        
    k = cv2.waitKey(1)
    time.sleep(0.015)
    if k == ord('s'):
        break
    cv2.imshow('result',f)
    cv2.moveWindow('result',45,10)
    
cv2.destroyAllWindows()
