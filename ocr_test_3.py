import cv2
from algorithm.object_detector import YOLOv7
import threading as th
import time

yolov7 = YOLOv7()
yolov7.set(ocr_classes=['truck','bus','car'])
yolov7.load('coco.weights', classes='coco.yaml', device='cpu')

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

def ocrResult(f):
    global detections, busy
    busy=1
    image=f[100:500,400:1500]
    image=cv2.resize(image,(2000,700))
    detections = yolov7.detect(image)
    #cv2.imshow('detect',image)
    for n in detections:
        if ('car' in n['class']) or ('truck' in n['class']) or ('bus' in n['class']):
            result(detections)
    busy=0
       
def result(data):
    for n in data:
        d=''
        if 'text' in n:
            d=n['text']
        print('%s\t%s\t%s'%(n['class'],int(n['confidence']*100),d)) 
    
ip='192.168.100.69'
#ip='192.168.100.28'
src=cam(ip)

tmr=time.time()+1
busy=0
while 1:
    r,f=src.read()
    if not(r):
        continue
    if time.time()>tmr and not(busy):
        tmr=time.time()+1
        ocrResult(f)
    f=cv2.resize(f,(600,400))
    cv2.imshow('image',f)
    cv2.moveWindow('image',1,1)   
    k = cv2.waitKey(1)
    if k == ord('s'):
        break
    
cv2.destroyAllWindows()
