#!/usr/bin/python
# -*- coding: utf-8 -*-
from pprint import pprint
import json
from bson.objectid import ObjectId
from pymongo import MongoClient
from PIL import Image
import gridfs, os
import cv2
import numpy as np
from collections import defaultdict

#/ creating connections for communicating with Mongo DB
client = MongoClient('localhost', 27017)
db = client.knots
fs = gridfs.GridFS(db)
fm = db.marked_up_image
data_dict = defaultdict(list)
data= []

for gridout in fm.find({"users_polygons.polygons.type": {"$exists": True}}):

    images = []
    plist = gridout['users_polygons']
    for usrpoly in plist:
        poly = usrpoly['polygons']

        for p in poly:
            #pprint (p)
            points = p['points']
            if p['type'] != 'surface':
                tmp = []
                for point in points:
                    #pprint (point)
                    tmp.append((point['x'], point['y']))
                cor = np.asarray([tmp], dtype=np.int32)
                data_dict[p['type']].append({'image': gridout['image'], 'coordinates': cor})


keys = []

for i, key in enumerate(data_dict.keys()):
    print(i, key)
    keys.append((i, key))
    data_dict[i] = data_dict[key].copy()
    del data_dict[key]


input("Press Enter to continue...")

with open('keys.txt', 'w') as fk:
    fk.write(str(keys))

dd_train = {}
dd_test = {}

# with open('defs.txt', 'r') as d:
#     data_dict = json.load(d)
# print (type(data_dict))


for key in data_dict.keys():
    length = len(data_dict[key])
    length_train = int(length // (10 / 7))
    #print(key, len(dd[key]))
    dd_train[key] = data_dict[key][:length_train]
    dd_test[key] = data_dict[key][length_train+1:]

def_path= ''

for key in dd_train.keys():
    path = str(key)
    for val in dd_train[key]:
        try:
            with open('tmp.png', 'wb') as fi:
                fi.write(fs.get( ObjectId(val['image']) ).read() )
                img = cv2.imread('tmp.png', -1)

                mask = np.zeros(img.shape, dtype=np.uint8)
                roi_corners = val['coordinates']

                channel_count = img.shape[2]
                ignore_mask_color = (255,) * channel_count
                try:
                    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                except:
                    continue
                masked_image = cv2.bitwise_and(img, mask)

                if not os.path.exists('train/' + path):
                    os.makedirs('train/' + path)

                cv2.imwrite(os.path.join('train/' + path, 'defect' + str(ObjectId(val['image']))
                                         + '.png'), masked_image)
                def_path += '/train/' + path + '/' 'defect' + str(ObjectId(val['image'])) + '.png' + ' ' + path + '\n'

        except StopIteration:
            print("zero cursor")

with open('train.txt', 'w') as f:
    f.write(def_path)

def_path = ''

for key in dd_test.keys():
    path = str(key)
    for val in dd_test[key]:
        try:
            with open('tmp.png', 'wb') as fi:
                fi.write(fs.get(ObjectId(val['image'])).read())
                img = cv2.imread('tmp.png', -1)

                mask = np.zeros(img.shape, dtype=np.uint8)
                roi_corners = val['coordinates']

                channel_count = img.shape[2]
                ignore_mask_color = (255,) * channel_count
                try:
                    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                except:
                    continue
                masked_image = cv2.bitwise_and(img, mask)

                if not os.path.exists('test/' + path):
                    os.makedirs('test/' + path)

                cv2.imwrite(os.path.join('test/' + path, 'defect' + str(ObjectId(val['image']))
                                         + '.png'), masked_image)

                def_path += '/test/' + path + '/' 'defect' + str(ObjectId(val['image'])) + '.png' + ' ' + path + '\n'

        except StopIteration:
            print("zero cursor")

with open('test.txt', 'w') as f:
    f.write(def_path)

#
# for defect in data:
#     if defect['type'] != 'surface':
#         try:
#             with open('tmp.png', 'wb') as fi:
#                 fi.write(fs.get( ObjectId(defect['image']) ).read() )
#                 img = cv2.imread('tmp.png', -1)
#
#                 mask = np.zeros(img.shape, dtype=np.uint8)
#                 roi_corners = defect['coordinates']
#
#                 channel_count = img.shape[2]
#                 ignore_mask_color = (255,) * channel_count
#
#                 cv2.fillPoly(mask, roi_corners, ignore_mask_color)
#
#                 masked_image = cv2.bitwise_and(img, mask)
#                 if not os.path.exists('images/surface/images'):
#                     os.makedirs('images/surface/images')
#                 if not os.path.exists('images/surface/defect'):
#                     os.makedirs('images/surface/defect')
#                 cv2.imwrite(os.path.join('images/surface/images',  str(ObjectId(defect['image'])) + '.png'), img)
#                 cv2.imwrite(os.path.join('images/surface/defect', 'defect' + str(ObjectId(defect['image'])) + '.png'), masked_image)
#
#
#
#         except StopIteration:
#             print("zero cursor")
#             # data.append(data_dict)