from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2
import time
import threading as th

yolov7 = YOLOv7()
yolov7.set(ocr_classes=['truck','bus','car'])
yolov7.load('coco.weights', classes='coco.yaml', device='cpu')

def ocrResult(image):
    detections = yolov7.detect(image)
    #detected_image = draw(image, detections)
    cv2.imshow('detect',image)
    d=''
    for n in detections:
        if 'text' in n:
            d=(n['text'])
            conf=int(n['confidence']*100)
            ok=''
            if conf>75:
                ok=fPlate(d)
            elif (conf>35) and ('truck' in n['class']):
                ok=fPlate(d)
            if ok:
                cv2.putText(frame,
                            ok,
                            (30,70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255,0,0),
                            2,
                            cv2.LINE_AA)
                cv2.imwrite('./img/%s_%s.png'%(time.strftime('%H:%M:%S'),
                                                             ok),
                            frame)
            if d:
                print('%s\t%s\t%s %s [%s]'%(n['class'],
                                    int(n['confidence']*100),
                                    time.strftime('%H:%M:%S'),
                                    ok,d
                                    ))
     

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
def fPlate(x):
    x+=';'
    x=x.split(';')
    for n in x:
        if 5<len(n)<10:
            r=''
            for i in n:
                if (64<ord(i)<91) or (47<ord(i)<58):
                    r+=i
            if (len(r)==6) and (47<ord(r[-1])<58) and (64<ord(r[1])<91):
                return r
    return ''
            

ip='192.168.100.69'
#ip='192.168.100.44'
src=cam(ip)
tma=time.time()+3

while 1:
    r,frame=src.read()
    if time.time()>tma:
        tma=time.time()+1
        image = frame[170:550,300:2000]
        th._start_new_thread(ocrResult,(image,))
    k = cv2.waitKey(1)
    if k == ord('s'):
        break
    #f=frame[170:550,300:2000]
    #f=cv2.resize(image,(800,600))
    cv2.imshow('result',frame)
    #cv2.moveWindow('result',5,5)
    
cv2.destroyAllWindows()
