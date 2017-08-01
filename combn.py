import cv2
import dlib
import numpy 
import matplotlib.pyplot as plt
from numpy import linalg as LA



image=cv2.imread('27.jpg',1)
image1=cv2.imread('72.jpg',1)

PREDICTOR_PATH = "/home/arundhati/cv/dlib_project/shape_predictor_68_face_landmarks.dat/data"

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)	
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im,landmarks):
        im=im.copy()
        for idx,point in enumerate(landmarks):
                pos=(point[0,0],point[0,1])
                cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0,0,255))
                #cv2.circle(im,pos,3,color=(0,255,255))
        return im

def calculate(landmarks):
	c=[]
	co=0
        for point in (landmarks):
                pos=(point[0,0],point[0,1])
		co=co+1
		c.append(pos)
	print co
	#print c

	combn=[]
	for i in range(0,68,1):
		for j in range(0,68,1):
			x1,x2,y1,y2=scale(c[i][0],c[j][0],c[i][1],c[j][1])
			#d=(((c[i][0]-c[j][0])**2)+((c[i][1]-c[j][1])**2))
			d=(((x1-x2)**2)+((y1-y2)**2))
			s=numpy.sqrt(d)
			combn.append(s)
	print len(combn)
	print combn[0],combn[1]
	

def scale(x1,x2,y1,y2):
    lmx=LA.norm(x1)
    rmx=LA.norm(x2)
    lmy=LA.norm(y1)
    rmy=LA.norm(y2)
    return lmx,rmx,lmy,rmy
	


landmarks=get_landmarks(image)
print landmarks
landmarks1=get_landmarks(image1)
image_with_landmarks=annotate_landmarks(image,landmarks)
image_with_landmarks1=annotate_landmarks(image1,landmarks1)

cv2.imshow("image",image_with_landmarks)
cv2.waitKey(0)
cv2.imshow("image",image_with_landmarks1)
cv2.waitKey(0)
while(1):
	if  cv2.waitKey(0) & 0xFF == 27:
        	break
cv2.destroyAllWindows()


calculate(landmarks)
calculate(landmarks1)













