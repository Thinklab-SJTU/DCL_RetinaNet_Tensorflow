# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
OBB task:
iou threshold: 0.5
classname: small-vehicle
npos num: 5090
ap:  0.48448423252573813
classname: ship
npos num: 9886
ap:  0.8558483622149499
classname: plane
npos num: 2673
ap:  0.8962403902596927
classname: large-vehicle
npos num: 4293
ap:  0.7655353635381923
classname: helicopter
npos num: 72
ap:  0.33873388441995184
classname: harbor
npos num: 2065
ap:  0.6142704774599544
map: 0.6591854517364132
classaps:  [48.44842325 85.58483622 89.62403903 76.55353635 33.87338844 61.42704775]

AP50:95: [0.6591854517364132, 0.6337045704100505, 0.601037302774938, 0.5439552301875512, 0.4861559177256463,
          0.3879971404977413, 0.25908752720368566, 0.14314668041497244, 0.05059363339801165, 0.0014117567590187534]
mmAP: 0.3766275211108029

OHD task:
iou threshold: 0.5
classname: small-vehicle
npos num: 5090
ap:0.26592393142840554, ha:0.6014925358171828
classname: ship
npos num: 9886
ap:0.4757000858277587, ha:0.6838502666482885
classname: plane
npos num: 2673
ap:0.599343688640907, ha:0.7443403561891476
classname: large-vehicle
npos num: 4293
ap:0.35320201924364, ha:0.5778635763761555
classname: helicopter
npos num: 72
ap:0.175276963180189, ha:0.49056594517623675
classname: harbor
npos num: 2065
ap:0.4129432440713498, ha:0.7665995924440001
map:0.380398322065375, mha:0.6441187121085018
classaps:[26.59239314 47.57000858 59.93436886 35.32020192 17.52769632 41.29432441], classhas:[60.14925358 68.38502666 74.43403562 57.78635764 49.05659452 76.65995924]

AP50:95: [0.380398322065375, 0.36990593991261894, 0.34682245559686486, 0.3299012031633337, 0.2906761435543597,
          0.2485968338563105, 0.1725739242074585, 0.11221284686524324, 0.044103676976843496, 0.001404915370226375]
mmAP: 0.22965962615686344
HA50:95: [0.6441187121085018, 0.6445845874623773, 0.6398003488798905, 0.6564484168826371, 0.6519379492440414,
          0.6517355174217961, 0.6709016308200306, 0.6739228151529306, 0.6294422142250025, 0.5487629887892586]
mmHA: 0.6411655180986465
"""

# ------------------------------------------------
VERSION = 'RetinaNet_OHD-SJTU-ALL_R3Det_CSL_OHDet_2x_20200818'
NET_NAME = 'resnet101_v1d'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "1,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 20000 * 2

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_CLS_WEIGHT = 0.5
HEAD_CLS_WEIGHT = 0.1
USE_IOU_FACTOR = True

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4 * BATCH_SIZE * NUM_GPU
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 8.0 * SAVE_WEIGHTS_INTE)


# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'OHD-SJTU-ALL-HEAD-600'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 800
CLASS_NUM = 6
LABEL_TYPE = 0
RADUIUS = 4
OMEGA = 1

IMG_ROTATE = False
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# --------------------------------------------- Network_config
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4
USE_GN = False
NUM_SUBNET_CONV = 4
NUM_REFINE_STAGE = 1
USE_RELU = False
FPN_CHANNEL = 256

# ---------------------------------------------Anchor config
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 5.]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True
METHOD = 'H'
USE_ANGLE_COND = False
ANGLE_RANGE = 90

# --------------------------------------------RPN config
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

NMS = True
NMS_IOU_THRESHOLD = 0.1
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

# --------------------------------------------MASK config
USE_SUPERVISED_MASK = False
MASK_TYPE = 'r'  # r or h
BINARY_MASK = False
SIGMOID_ON_DOT = False
MASK_ACT_FET = True  # weather use mask generate 256 channels to dot feat.
GENERATE_MASK_LIST = ["P3", "P4", "P5", "P6", "P7"]
ADDITION_LAYERS = [4, 4, 3, 2, 2]  # add 4 layer to generate P2_mask, 2 layer to generate P3_mask
ENLAEGE_RF_LIST = ["P3", "P4", "P5", "P6", "P7"]
SUPERVISED_MASK_LOSS_WEIGHT = 1.0
