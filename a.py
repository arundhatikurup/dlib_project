import numpy

class TooManyFaces(Exception):
	pass

class NoFaces(Exception):
	pass

def annotate_landmarks(im,landmarks):
	im=im.copy()
	for idx,point in enumerate(landmarks):
		pos=(point[0,0],point[0,1])
	cv2.putText[im,str(idx),pos,fontFace=cv2,FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0,0,255))
	cv2.circle(im,pos,3,color=(0,255,255))
	return im

image=cv2.imread('27.jpg')
landmarks=get_landmarks(image)
image_with_landmarks=annotate_landmarks(image,landmarks)
