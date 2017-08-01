import cv2 , os
import dlib
import numpy 
import matplotlib.pyplot as plt
from PIL import Image
from numpy import linalg as LA
import time
from sklearn.ensemble import ExtraTreesClassifier




start_time = time.time()


PREDICTOR_PATH = "/home/arundhati/cv/dlib_project/shape_predictor_68_face_landmarks.dat/data"

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    #cv2.imshow("image",im)
    #cv2.waitKey(25)
    try:
    	rects = detector(im, 1)	
    	a=numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])		
    except IndexError :
	pass
	
    else :
	return a




def scale(x1,x2,y1,y2):
	v1=[x1,x2]
	v2=[y1,y2]
	mx=LA.norm(v1)
	my=LA.norm(v2)
    	#print mx,my
    	a=x1/mx
    	b=x2/mx
    	c=y1/my
    	d=y2/my
    	#print a,b,c,d
    	return a,b,c,d

def calculate(landmarks):
	c=[]
	#co=0
        for point in (landmarks):
                pos=(point[0,0],point[0,1])
		#co=co+1
		c.append(pos)
	#print co
	#print c

	combn=[]
	for i in range(0,68,1):
		for j in range(0,68,1):
			#print c[i][0],c[j][0],c[i][1],c[j][1]
			x1,x2,y1,y2=scale(c[i][0],c[j][0],c[i][1],c[j][1])
			#print x1,x2,y1,y2
			#d=(((c[i][0]-c[j][0])**2)+((c[i][1]-c[j][1])**2))
			d=(((x1-x2)**2)+((y1-y2)**2))
			s=numpy.sqrt(d)
			combn.append(s)
	return combn
	#print len(combn)-each is 68*68
	#print combn[10],combn[100],combn[23],combn[37]
	


path = '/home/arundhati/cv/dlib_project/faces' 



image_paths = [os.path.join(path, f) for f in os.listdir(path)]


p=[]
final=[]
feature=[]
for image_path in image_paths:
	image_pil = Image.open(image_path).convert('L')
	image = numpy.array(image_pil, 'uint8')
	#print image_path
	landmarks=get_landmarks(image)
	feature.append(calculate(landmarks))	
	nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
	p.append(nbr)
	total=[]
	for point in (landmarks):
		pos=(point[0,0],point[0,1])	
		value=[pos,nbr]
		total.append(value)
	final.append(total)

print "final",len(final) #[[(x,y),class]*68]]*162
print "label",len(p)#162 class labels
print "feature",len(feature) 

#print feature[0]
#print feature #162 elements of (68*68)elments each 

print("--- %s seconds ---" % (time.time() - start_time))

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(feature, p)
importances = forest.feature_importances_

std = numpy.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = numpy.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(feature[1])):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(feature[1])), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(len(feature[1])), indices)
plt.xlim([-1, len(feature[1])])
plt.show()



