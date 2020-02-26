from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os
import tensorflow as tf
from keras import backend as K
from keras_frcnn import config, data_generators
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
from keras_frcnn.pascal_voc_parser import get_data

from keras_frcnn import vgg
from keras_frcnn import rpn
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')

# pass the settings from the command line, and persist them in the config object
C = config.Config()

# parser
all_imgs, classes_count, class_mapping = get_data(options.train_path)

# bg
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = 'config.pickle'

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
val_imgs = [s for s in all_imgs if s['imageset'] == 'val']
test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))
print('Num test samples {}'.format(len(test_imgs)))

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
C.model_path = 'model.hdf5'
C.num_rois = 32

# Defined Anchor
data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')

input_shape_img = (None, None, 3)

# input placeholder
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# base network(feature extractor)
shared_layers = vgg.nn_base(img_input, trainable=True)

# RPN bsae from vgg layer
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = rpn.rpn(shared_layers, num_anchors)

# detection network
classifier = vgg.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 1000
num_epochs = int(options.num_epochs)
iter_num = 0
train_step = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

print('Starting training')

for epoch_num in range(num_epochs):
    
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        