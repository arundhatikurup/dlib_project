import os
import numpy 
import cv2
from matplotlib import pyplot as plt
from PIL import Image

face_cascade = cv2.CascadeClassifier('/home/arundhati/cv/dlib_project/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/arundhati/cv/dlib_project/haarcascade_eye.xml')

image_rootDir="/home/arundhati/cv/dlib_project/orl_faces"
for dirName, subdirList, fileList in os.walk(image_rootDir):
        for fname in fileList:
            if fname.endswith(('.pgm')):
                #p1=[]
                label=os.path.basename(dirName)
                image_path = os.path.join(dirName, fname)
                image_pil = Image.open(image_path).convert('L')
                image = numpy.array(image_pil, 'uint8')
                cv2.imshow("image",image)
                cv2.waitKey(25)
                faces = face_cascade.detectMultiScale(image,1.1, 5)
  
                try:
                    for (x,y,w,h) in faces:
                        roi_gray = image[y:y+h, x:x+w]
                        cv2.imwrite("/home/arundhati/cv/dlib_project/eyes/" + str(i+1) + ".jpg",roi_gray)
                        eyes = eye_cascade.detectMultiScale(roi_gray)
                        for (ex,ey,ew,eh) in eyes:
                            eye_gray=image[ey:ey+eh,ex:ex+ew]
                            cv2.imwrite("/home/arundhati/cv/dlib_project/eyes/" +fname + ".jpg",eye_gray)
                except TypeError: 
                    print fname
                    continue
