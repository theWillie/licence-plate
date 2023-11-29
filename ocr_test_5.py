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

def ocrResult(fr):
    global detections, busy
    busy=1
    image=fr[100:500,400:1500]
    image=cv2.resize(image,(2000,700))
    detections = yolov7.detect(image)
    #cv2.imshow('detect',image)
    for n in detections:
        if ('car' in n['class']) or ('truck' in n['class']) or ('bus' in n['class']):
            r=result(detections)
            if r:
                cv2.putText(fr,
                            r,
                            (30,70),
                            cv2.FONT_HERSHEY_SIMPLEX,#font
                            2,#scale
                            (255,0,0),#color
                            3,#thic
                            cv2.LINE_AA)
                cv2.imwrite('./img/%s_%s.png'%(time.strftime('%d_%H:%M:%S'),
                                                             r),
                            fr)
    busy=0
       
def result(data):
    for n in data:
        d=''
        conf=int(n['confidence']*100)
        clss=n['class']
        if 'text' in n:
            d=n['text']
            if conf>30 and 'truck' in clss:
                ok=True
            elif conf>40 and 'car' in clss:
                ok=True
            elif conf>30 and 'bus' in clss:
                ok=True
            else:
                ok=False
            if ok:
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
backG=None
while 1:
    r,f=src.read()
    if not(r):
        continue
    img=f[100:500,400:1500]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    if backG is None:
        backG=gray
        continue
    diff=cv2.absdiff(backG,gray)
    backG=gray.copy()
    thres=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
    thres=cv2.dilate(thres,None,iterations=2)
    cnts,h=cv2.findContours(thres.copy(),
                          cv2.RETR_TREE,
                          cv2.CHAIN_APPROX_SIMPLE)
    if time.time()>tmr and not(busy):
        tmr=time.time()+0.5
        for n in cnts:
            a=cv2.contourArea(n)
            if (a>5000) and not(busy):
                th._start_new_thread(ocrResult,(f,))
    f=cv2.resize(f,(600,400))
    cv2.imshow('image',f)
    cv2.moveWindow('image',1,1)   
    k = cv2.waitKey(1)
    if k == ord('s'):
        break
    
cv2.destroyAllWindows()
