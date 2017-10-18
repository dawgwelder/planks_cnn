import os, sys
from datetime import datetime, date, time
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import localization as lc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

'''image'''
#img_path = 'img/knot_sound/img57d94ac898177fc2b2b341a4.png' #massive knot
img_path = 'img/knot_encased/img57d94ac998177fc2b2b343fc.png' #tar and massive knot

#img_path = 'img/knot_defect/img57d94ac898177fc2b2b3422e.png' # flawed knot

#img_path = 'img/darken/img57d9a0ceb12d55004c4c9666.png' # some mechs
#
# img_path = 'img/surface/img57d9a0c4b12d55004c4c9229.png' # flawed
#img_path = 'img/knot_encased/img57d94aca98177fc2b2b34724.png' # ###corrupted and massive
#img_path = 'planks/knot_decay/images/57d9a103b12d55004c4ca4af.png' #
#img_path = 'planks/knot_decay/images/5787a47651208b00287fed8c.png' #
#img_path = 'planks/knot_sound/images/5787a46e51208b00287feb4a.png'
#img_path = 'planks/surface/images/57d9a0cbb12d55004c4c952c.png' #
#img_path = 'planks/surface/images/57d9a0d7b12d55004c4c991a.png' #
#img_path = 'planks/surface/images/57d9a112b12d55004c4ca8fb.png' #
#img_path = 'planks/knot_sound/images/57d9a103b12d55004c4ca4af.png' #
#img_path = 'planks/knot_sound/images/5787a50e51208b00288009d0.png' #
#img_path = 'planks/knot_decay/images/57d9a0cfb12d55004c4c968a.png' #
#img_path = 'planks/knot_decay/images/57d9a0e7b12d55004c4c9e0b.png' #
#img_path = 'planks/knot_decay/images/57d9a0ecb12d55004c4c9f7c.png' #
#img_path = 'planks/knot_decay/images/57d9a0eeb12d55004c4ca023.png' #
#img_path = 'planks/knot_decray/images/57d9a120b12d55004c4cad3d.png' #
#img_path = 'planks/knot_decay/images/57d9a134b12d55004c4cb1ef.png' #
#img_path = 'planks/knot_decay/images/57d9a137b12d55004c4cb26c.png' # flawed
#img_path = 'planks/tar/images/5787a48d51208b00287ff250.png' #
#img_path = 'planks/tar/images/57d9a127b12d55004c4caf02.png' #
#img_path = 'planks/tar/images/57d9a10bb12d55004c4ca70d.png' #

#img_path = 'img/darken/img57d94ac998177f5787a50651208b00288008772b2b345f9.png' # massive knot
#img_path = 'img/knot_sound/img57d94ac898177fc2b2b34119.png' # massive knot
#img_path = 'img/knot_sound/img57d94ac898177fc2b2b3417b.png' # massive knot

#img_path = 'defects/pic014.png'


'''path'''

'''images'''
img = cv2.imread(img_path)
img = lc.cut_board(img)
contours = lc.find_contours(img)
mechs = lc.mech(img)
# for m in mechs:
#     img = cv2.circle(img, (m[0], m[1]), 12, (255,0,0), 1, cv2.LINE_AA)
#     print (m)
cn1 = []

cn1 = contours

print (cn1)
cn1.extend(mechs)
# con = contours[1]
# cn = []
# for cnt in (contours[2]):
#     for idx, hier in enumerate(cnt):
#         if hier[3] == -1:
#             cn.append(con[idx])
#
# cn1 = []
# height, width, channels = img.shape
# arp = width * height
# # print(arp)
# # print('imgsize', width,height)
# for cnt in cn:
#     ar = cv2.contourArea(cnt)
#     per = cv2.arcLength(cnt, True)
#     # print (per)
#     # M = cv2.moments(cnt)
#     # cx = int(M['m10'] / M['m00'])
#     # cy = int(M['m01'] / M['m00'])
#     # print(abs(width-cx) )
#     # if  per < width*0.5:
#     if ar + per * 3 < arp and ar > 70:
#         cn1.append(cnt)
#         x = ar + per * 3
#         # print(x)
vis = img.copy()

channel_count = img.shape[2]
ignore_mask_color = (255,) * channel_count
defects = []



'''______'''
learning_rate = 0.01
num_epochs = 10
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 11
train_layers = ['fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "C:/TFLogs/tmp/finetune_alexnet/defects"
checkpoint_path = "C:/TFLogs/tmp/finetune_alexnet/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()


def get_img(img_path):
    """
    This function gets the next n ( = batch_size) images from the path list
    and labels and loads the images into them into memory
    """
    # Get next batch of image (path) and labels

    # Read images
    scale_size = (227, 227)
    images = np.ndarray([128, scale_size[0], scale_size[1], 3])
    img = cv2.imread(img_path)

    # rescale image
    img = cv2.resize(img, (scale_size[0], scale_size[0]))  # second scale_size was [0]
    img = img.astype(np.float32)

    # subtract mean
    # img -= self.mean

    images[0] = img
    return images

def get_images(defects):

    # Read images
    scale_size = (227, 227)
    images = np.ndarray([128, scale_size[0], scale_size[1], 3])
    for i, img in enumerate(defects):


        # rescale image
        img = cv2.resize(img, (scale_size[0], scale_size[0]))  # second scale_size was [0]
        img = img.astype(np.float32)

        # subtract mean
        # img -= self.mean

        images[i] = img
    return images

classes = []

with tf.Session(config=config) as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    model.load_initial_weights(sess)
    saver.restore(sess, 'C:/TFLogs/tmp/finetune_alexnet/model_epoch616.ckpt')
    t = datetime.now()
    for idx, cnt in enumerate(cn1):
        hull = cv2.convexHull(cnt)

        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [hull], ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        defects.append(masked_image)

    images = get_images(defects)

    classification = sess.run(score, {x: images, keep_prob: 1.})

    # norm = tf.sqrt(tf.reduce_sum(tf.square(classification[0]), 0, keep_dims=True))
    # res = x / norm
    res = tf.nn.l2_normalize(classification, 1, epsilon=1e-12, name=None)
    for idx, item in enumerate(defects):
        label = sess.run(tf.argmax(res[idx]))
        preds = sess.run(res)[idx]

        keys = {0: 'mechanical defect', 1: 'rested knot', 2: 'corrupted knot',
                3: 'pin knot', 4: 'unknown defect', 5: 'massive knot',
                6: 'tar', 7: 'crack', 8: 'flawed knot', 9: 'fallen knot',
                10: 'darken'}
        classes.append({'label': keys[label], 'color': label})

    patch = []
    sortcols = []

    cols = {0: 'red', 1: 'green', 2: 'blue',
            3: 'yellow', 4: 'teal', 5: 'cyan',
            6: 'navy', 7: 'darkmagenta', 8: 'orange',
            9: 'tan', 10: 'azure'}

    for idx, cnt in enumerate(cn1):
        hull = cv2.convexHull(cnt)
        name = classes[idx]['label']
        col = cols[int(classes[idx]['color'])]
        color = colors.to_rgb(col)
        color = [int(i*255) for i in color]
        color.append(30)
        color = tuple(color)

        #cv2.fillConvexPoly(vis, hull, color, lineType=cv2.LINE_AA)
        cv2.drawContours(vis, [hull], -1, color, 1, lineType=cv2.LINE_AA)
        if col not in sortcols:
            patch.append(mpatches.Patch(color=col, label=name))
            sortcols.append(col)


    patch = list(set(patch))
    plt.imshow(vis)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, handles=patch, ncol=10, mode="expand", borderaxespad=0.)
    plt.xticks([]), plt.yticks([])

    plt.show()
    print (datetime.now() - t)
    sess.close()

