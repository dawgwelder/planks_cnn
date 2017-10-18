import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.externals import joblib

from itertools import cycle

clf = joblib.load('planks.pkl')
xy = []
img = cv2.imread('cl_test2.png')
# img = img[180:260, 200:320]
img = img[120:450, :]
vis =img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SURF_create()
kp, dsc = sift.detectAndCompute(gray, None)
img = cv2.drawKeypoints(gray, kp, vis)
# cv2.imshow('', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
for kpt in kp:
    x, y = kpt.pt
    xy.append([x,y])
xy = np.asarray(xy)

af = AffinityPropagation(preference=-50000).fit(xy)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)
plt.close('all')
plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k in range(n_clusters_):
#     class_members = labels == k
#     xy = np.delete(xy, class_members, None)
print (labels)
print (len(labels))
print (len(xy))

restricted_labels = []
for k in range(n_clusters_):
    # print(k , col)
    class_members = labels == k

    #cluster_center = xy[cluster_centers_indices[k]]
    print (len(xy[class_members]))
    if len(xy[class_members]) <= 10:
        restricted_labels.append(k)

for k, col in zip(range(n_clusters_), colors):
    # print(k , col)
    if k in restricted_labels:
        continue
    class_members = labels == k
    # class_members = asdasd(k)
    # print(class_members)
    cluster_center = xy[cluster_centers_indices[k]]
    # plt.plot(xy[class_members, 0], xy[class_members, 1], col + '.')
    # plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #          markeredgecolor='k', markersize=14)
    res = clf.predict(dsc[class_members])
    print(res)
    hulls = [cv2.convexHull(xy[class_members].astype(np.float32))]
    cv2.polylines(vis, np.int32(hulls), 1, (0, 255, 0))
    # for x in xy[class_members]:
    #     plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
# print(len(labels))
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

#img = cv2.drawKeypoints(gray, kp, img)  # 625 210 750 300
cv2.imshow('', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()





