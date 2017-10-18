
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

pic_pth = 'img/knot_encased/img57d94aca98177fc2b2b34724.png'

img = cv2.imread(pic_pth)
img = img[80: 480,]
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
kernel = np.ones((5,5),np.uint8)
#sobely = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=-1)

#for i in range(5):
#	gray = cv2.pyrUp(gray)
#for i in range(3):
#	gray = cv2.pyrDown(gray)



#img = cv2.erode(img,np.ones((3,3),np.uint8),iterations = 2)
#img = cv2.equalizeHist(img)


#img = cv2.inRange(img, 0, 15)
#for i in range(4):
#	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
titles = ['1', '2', '3', '4']
clahe = cv2.createCLAHE(clipLimit=200.0, tileGridSize=(15,15))
gray1 = clahe.apply(gray)

#gray1 = cv2.bilateralFilter(gray1,9,75,75)
sobel = cv2.Sobel(gray1,cv2.CV_64F,0,1,ksize=5)

gray1 = gray / gray1

#sobel = cv2.Sobel(gray1,cv2.CV_64F,1,0,ksize=5)
#gray1 = cv2.pyrDown(gray1)
#gray1 = cv2.pyrUp(gray1)
#gray1 = cv2.resize(gray1, (gray.shape[1], gray.shape[0])) 
#plt.hist(gray.ravel(),10,[0,256]); plt.show()

#ret3,gray1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#gray1 = cv2.inRange(gray1, 0, 60)
gray1 = cv2.inRange(gray1, 0, 10)
gray1= cv2.morphologyEx(gray1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
gray1= cv2.morphologyEx(gray1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
gray1= cv2.morphologyEx(gray1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
plt.hist(gray1.ravel(),10,[0,256]); plt.show()
plt.xlim([0, 256])

gray1 = cv2.pyrDown(gray1)
gray1 = cv2.pyrDown(gray1)
gray1 = cv2.pyrDown(gray1)

gray1 = cv2.pyrUp(gray1)
gray1 = cv2.pyrUp(gray1)
gray1 = cv2.pyrUp(gray1)
plt.hist(gray1.ravel(),10,[0,256]); plt.show()
plt.xlim([0, 256])
morphed = gray1 
#for i in range(3):
	#morphed = cv2.morphologyEx(morphed, cv2.MORPH_GRADIENT, kernel)
	#morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

#plt.hist(gray1.ravel(),10,[0,256]); plt.show()
gray1 = cv2.inRange(gray1, 0, 210)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
#contours = cv2.findContours(np.uint8(morphed), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(gray,  contours[1], -1, (0, 255, 0), 1)
#morphed = cv2.erode(morphed,kernel,iterations = 1)
#morphed = cv2.dilate(morphed,kernel,iterations = 3)
#cv2.bitwise_or(gray1, gray1, mask = np.uint8(morphed)
gray1 = cv2.resize(gray1, (gray.shape[1], gray.shape[0]))
cv2.imshow('', morphed)
cv2.waitKey(0)
cv2.destroyAllWindows()

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 215    # the graylevel of images
params.maxThreshold = 226
#params.filterByColor = True
#params.blobColor = 0

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(morphed)
xy = []
for kpt in keypoints:
    x, y = kpt.pt
    xy.append([x,y])

xy = np.asarray(xy)
print (xy)
for p in xy:
    x=p[0]
    y=p[1]
    out = cv2.circle(img, (int(x), int(y)), 14, (255, 0, 255), 1, lineType=cv2.LINE_AA)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
print(keypoints)
#out = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

res = cv2.bitwise_and(gray, gray1, img)

cv2.imwrite('mech.png', out)
images = [gray, gray1, morphed, out] #sobely, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()