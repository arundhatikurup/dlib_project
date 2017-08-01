import cv2
import dlib
import numpy 
import matplotlib.pyplot as plt



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

	rx=[]
	ry=[]
	lx=[]
	ly=[]

	for i in range(36,42,1):
		rx.append(c[i][0])
		ry.append(c[i][1])
	#print rx,ry

	for j in range(42,48,1):
		lx.append(c[j][0])
		ly.append(c[j][1])
	#print lx,ly

	x1=numpy.mean(lx)
	x2=numpy.mean(rx)
	y1=numpy.mean(ly)
	y2=numpy.mean(ry)

	print x1,x2,y1,y2
        
	d=(((x1-x2)**2)+((y1-y2)**2))
	s=numpy.sqrt(d)
	print s
	



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

#left= list(range(42, 48))
#right= list(range(36, 42))



calculate(landmarks)
calculate(landmarks1)













