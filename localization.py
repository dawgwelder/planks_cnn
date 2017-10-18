import cv2
import numpy as np
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot as plt

def cut_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    # th2 = cv2.inRange(sobely, 100, 255)
    sobely = cv2.Canny(gray, 150, 255, 10)
    kernel = np.ones((3, 3), np.uint8)
    sobely = cv2.dilate(sobely, kernel, iterations=5)

    lines = cv2.HoughLines(sobely, 1, np.pi / 2, 200)
    #print(len(lines))
    y_1 = 0
    y_2 = 10000
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            y_1 = max(y_1, y1)
            y_2 = min(y_2, y2)
            # cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
            # cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
    #rect = cv2.boundingRect(y1,y2)
    img = img[y_2+10:y_1-10, :]
    return img

def find_contours (img):
    av_pix = np.asarray([img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()])
    max_pix = np.asarray([img[..., 0].max(), img[..., 1].max(), img[..., 2].max()])
    min_pix = np.asarray([img[..., 0].min(), img[..., 1].min(), img[..., 2].min()])

    coeff = 0.15
    res = max_pix - min_pix
    # max_pix = max_pix - res*coeff
    min_pix *= 5
    # print(min_pix.dtype)
    #
    # print(max_pix.dtype)
    # np.array([255, 255, 255], np.uint8)
    # color = ('b', 'g', 'r')
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    #     plt.plot(histr, color=col)
    #     plt.xlim([0, 256])
    # plt.show()
    # mask = cv2.inRange(img, min_pix, max_pix)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histr = cv2.calcHist([gray], [0], None, [10], [0, 256])
    # plt.plot(histr)
    # plt.show()
    # plt.xlim([0, 256])
    # print(histr)
    persum = 0
    s =  np.sum(histr)
    # print(s)
    ind = 0
    for i, num in enumerate(histr):
        persum+=num/s
        if persum > 0.045:
            break
        ind = i

    ind += 1
    kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=3)
    # mask = cv2.Canny(gray, 0, 100, 10)
    mask = cv2.inRange(gray, 0, 26*ind)


    height, width, channels = img.shape
    line_width = 10
    line_color = (0, 0, 0)
    cv2.line(mask, (0, 0), (width, 0), line_color, line_width)
    cv2.line(mask, (0, height), (width, height), line_color, line_width)
    cv2.line(mask, (0, 0), (0, height), line_color, line_width)
    cv2.line(mask, (width, 0), (width, height), line_color, line_width)

    erosion = cv2.dilate(mask, kernel, iterations=5)
    er = erosion.copy()

    # r = cv2.erode(gray, kernel, iterations=10)
    # mask = cv2.Canny(r, 13*ind, 39*ind, 5)
    # d = cv2.dilate(mask, kernel, iterations=10)
    # cv2.bitwise_or(er, d, er)

    contours = cv2.findContours(er, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    con = contours[1]
    cn = []
    for cnt in (contours[2]):
        for idx, hier in enumerate(cnt):
            if hier[3] == -1:
                cn.append(con[idx])

    cn1 = []
    height, width, channels = img.shape
    arp = width * height
    # print(arp)
    # print('imgsize', width,height)
    for cnt in cn:
        ar = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt, True)
        # print (per)
        # M = cv2.moments(cnt)
        # cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])
        # print(abs(width-cx) )
        # if  per < width*0.5:
        if ar + per * 3 < arp and ar > 70:
            cn1.append(cnt)
            x = ar + per * 3
            # print(x)
    return cn1


def mech(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    # sobely = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=-1)

    # for i in range(5):
    #	gray = cv2.pyrUp(gray)
    # for i in range(3):
    #	gray = cv2.pyrDown(gray)



    # img = cv2.erode(img,np.ones((3,3),np.uint8),iterations = 2)
    # img = cv2.equalizeHist(img)


    # img = cv2.inRange(img, 0, 15)
    # for i in range(4):
    #	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    titles = ['1', '2', '3', '4']
    clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(15, 15))
    gray1 = clahe.apply(gray)

    # gray1 = cv2.bilateralFilter(gray1,9,75,75)
    sobel = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=5)
    gray1 = gray / gray1
    # sobel = cv2.Sobel(gray1,cv2.CV_64F,1,0,ksize=5)
    # gray1 = cv2.pyrDown(gray1)
    # gray1 = cv2.pyrUp(gray1)
    # gray1 = cv2.resize(gray1, (gray.shape[1], gray.shape[0]))
    # plt.hist(gray.ravel(),10,[0,256]); plt.show()

    # ret3,gray1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # gray1 = cv2.inRange(gray1, 0, 60)
    gray1 = cv2.inRange(gray1, 0, 10)
    gray1 = cv2.morphologyEx(gray1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    gray1 = cv2.morphologyEx(gray1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    gray1 = cv2.morphologyEx(gray1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    gray1 = cv2.pyrDown(gray1)
    gray1 = cv2.pyrDown(gray1)
    gray1 = cv2.pyrDown(gray1)
    gray1 = cv2.pyrUp(gray1)
    gray1 = cv2.pyrUp(gray1)
    gray1 = cv2.pyrUp(gray1)
    morphed = gray1
    # for i in range(3):
    # morphed = cv2.morphologyEx(morphed, cv2.MORPH_GRADIENT, kernel)
    # morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

    # plt.hist(gray1.ravel(),10,[0,256]); plt.show()
    gray1 = cv2.inRange(gray1, 0, 210)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # contours = cv2.findContours(np.uint8(morphed), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(gray,  contours[1], -1, (0, 255, 0), 1)
    # morphed = cv2.erode(morphed,kernel,iterations = 1)
    # morphed = cv2.dilate(morphed,kernel,iterations = 3)
    # cv2.bitwise_or(gray1, gray1, mask = np.uint8(morphed)
    gray1 = cv2.resize(gray1, (gray.shape[1], gray.shape[0]))

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 215  # the graylevel of images
    params.maxThreshold = 226
    # params.filterByColor = True
    # params.blobColor = 0

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(morphed)
    xy = []
    for kpt in keypoints:
        x, y = kpt.pt
        xy.append([int(x), int(y)])
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    mask = np.zeros(img.shape, dtype=np.uint8)
    for p in xy:
        mask = cv2.circle(mask, (p[0], p[1]), 12, ignore_mask_color, -1, cv2.LINE_AA)
    dil = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dil = cv2.cvtColor(dil, cv2.COLOR_BGR2GRAY)

    contours = cv2.findContours(dil, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    con = contours[1]
    cn = []

    for cnt in (contours[2]):
        for idx, hier in enumerate(cnt):
            if hier[3] == -1:
                cn.append(con[idx])
    return cn


def cl_find(img):
    xy = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SURF_create()

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=3)
    kp, dsc = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, kp, img)

    for kpt in kp:
        x, y = kpt.pt
        xy.append([x, y])
    xy = np.asarray(xy)

    af = AffinityPropagation(preference=-50000).fit(xy)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    restricted_labels = []
    for k in range(n_clusters_):

        class_members = labels == k

        print(len(xy[class_members]))
        if len(xy[class_members]) <= 10:
            restricted_labels.append(k)
    hulls = []

    for k in range(n_clusters_):

        if k in restricted_labels:
            continue
        class_members = labels == k

        hulls.append([cv2.convexHull(xy[class_members].astype(np.float32))])

    return hulls

# if __name__ == '__main__':
#     #pic_pth = 'img/mechanical/img57d94ac898177fc2b2b3435b.png'
#     #pic_pth = 'img/knot_sound/img57d94ac898177fc2b2b3417b.png'
#     pic_pth = 'img/knot_sound/img57d94ac898177fc2b2b341a4.png'
#     img = cv2.imread(pic_pth)
#     img = cut_board(img)
#     contours = find_contours(img)
#     mechcontours = mech(img)
#
#     # cn = contours
#     # cn1 = []
#     # height, width, channels = img.shape
#     # arp = width * height
#     # print(arp)
#     # print('imgsize', width,height)
#     # for cnt in cn:
#     #     ar = cv2.contourArea(cnt)
#     #     per = cv2.arcLength(cnt, True)
#     #     # print (per)
#     #     # M = cv2.moments(cnt)
#     #     # cx = int(M['m10'] / M['m00'])
#     #     # cy = int(M['m01'] / M['m00'])
#     #     # print(abs(width-cx) )
#     #     # if  per < width*0.5:
#     #     if ar + per * 3 < arp and ar > 70:
#     #         cn1.append(cnt)
#     #         x = ar + per * 3
#     #         # print(x)
#     cn1 = []
#     cn1.extend(contours)
#     # #cn1.extend(mechcontours[1])
#
#     vis = img.copy()
#
#     channel_count = img.shape[2]
#     ignore_mask_color = (255,) * channel_count
#
#     mask = np.zeros(img.shape, dtype=np.uint8)
#     for idx, cnt in enumerate(cn1):
#         hull = cv2.convexHull(cnt)
#
#         cv2.fillPoly(mask, [hull], ignore_mask_color)
#         masked_image = cv2.bitwise_and(img, mask)
#         cv2.drawContours(img, [hull], -1, (0, 255, 255), 1, lineType=cv2.LINE_AA)
#         #
#         # name = 'defects/' + str(idx) +'defect.png'
#         # cv2.imwrite(name, masked_image)
#
#     #cv2.drawContours(img, cn1, -1, (0, 255, 0), 2)
#
#     hulls = cl_find(vis)
#     for hull in hulls:
#         cv2.polylines(vis, np.int32(hull), 1, (0, 255, 0))
#     titles = ['method 1', 'method 2']
#     images = [img, vis]  # sobely]#, th1, th2, th3]
#     for i in range(2):
#         plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
#     plt.show()