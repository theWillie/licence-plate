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
            r=result(detections)
            if r:
                cv2.putText(f,
                            r,
                            (30,70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255,0,0),
                            2,
                            cv2.LINE_AA)
                cv2.imwrite('./img/%s_%s.png'%(time.strftime('%H:%M:%S'),
                                                             r),
                            f)
    busy=0
       
def result(data):
    for n in data:
        d=''
        if 'text' in n:
            d=n['text']
            if d:
                p=fPlate(d)
                print('%s %s\t%s\t%s'%(n['class'],
                                       time.strftime('%H:%M:%S'),
                                       int(n['confidence']*100),
                                       p))
                return p
    return False
                    
ant=''
def fPlate(x):
    global ant
    x+=';'
    x=x.split(';')
    for n in x:
        if 5<len(n)<10:
            r=''
            for i in n:
                if (64<ord(i)<91) or (47<ord(i)<58):
                    r+=i
            if (len(r)==6):
                if (47<ord(r[-1])<58) and (64<ord(r[1])<91):
                    if r!=ant:
                        ant=r
                        return r
    return ''

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
        tmr=time.time()+0.5
        #ocrResult(f)
        th._start_new_thread(ocrResult,(f,))
    f=cv2.resize(f,(600,400))
    cv2.imshow('image',f)
    cv2.moveWindow('image',1,1)   
    k = cv2.waitKey(1)
    if k == ord('s'):
        break
    
cv2.destroyAllWindows()
