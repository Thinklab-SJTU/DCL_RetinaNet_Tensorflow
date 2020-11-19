# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 64.

{'0.8': {'basketball-court': 0.33360746037488875, 'swimming-pool': 0.09090909090909091, 'ground-track-field': 0.24052910052910054, 'small-vehicle': 0.13559865425962703, 'harbor': 0.05515649050795279, 'mAP': 0.25212170035106235, 'helicopter': 0.10606060606060606, 'plane': 0.5268507124031996, 'large-vehicle': 0.08132054568913386, 'ship': 0.197594305903778, 'bridge': 0.024475524475524476, 'soccer-ball-field': 0.36451208706006144, 'tennis-court': 0.7853411931827885, 'storage-tank': 0.39282361062954446, 'baseball-diamond': 0.21816814320648675, 'roundabout': 0.2288779800741523},
'0.85': {'basketball-court': 0.11472938338609981, 'swimming-pool': 0.0048209366391184574, 'ground-track-field': 0.14248251748251747, 'small-vehicle': 0.045454545454545456, 'harbor': 0.009149940968122787, 'mAP': 0.1332488015102307, 'helicopter': 0.09090909090909091, 'plane': 0.29321050669533805, 'large-vehicle': 0.012735735293828881, 'ship': 0.10573539467320425, 'bridge': 0.005865102639296188, 'soccer-ball-field': 0.17193675889328064, 'tennis-court': 0.6544378352890649, 'storage-tank': 0.1909856544290416, 'baseball-diamond': 0.06313131313131314, 'roundabout': 0.09314730676959779},
'0.65': {'basketball-court': 0.5837869937032552, 'swimming-pool': 0.30190246383787867, 'ground-track-field': 0.4646930320387487, 'small-vehicle': 0.4907406433355843, 'harbor': 0.2840972731227815, 'mAP': 0.529427816191625, 'helicopter': 0.3615720703394156, 'plane': 0.8774900777407728, 'large-vehicle': 0.4153577566742698, 'ship': 0.6354828715853368, 'bridge': 0.20041802972706946, 'soccer-ball-field': 0.6102476342263835, 'tennis-court': 0.9012910101762504, 'storage-tank': 0.766233788923624, 'baseball-diamond': 0.5452780184773554, 'roundabout': 0.5028255789656493},
'0.95': {'basketball-court': 0.0, 'swimming-pool': 0.0, 'ground-track-field': 0.0, 'small-vehicle': 4.137873960359168e-05, 'harbor': 0.0, 'mAP': 0.0011534654688494692, 'helicopter': 0.0, 'plane': 0.00202020202020202, 'large-vehicle': 0.00011507479861910242, 'ship': 9.817396426467701e-05, 'bridge': 0.0, 'soccer-ball-field': 0.0, 'tennis-court': 0.010570824524312896, 'storage-tank': 0.00267379679144385, 'baseball-diamond': 0.0, 'roundabout': 0.0017825311942959},
'mmAP': 0.35999214231097615,
'0.9': {'basketball-court': 0.012987012987012986, 'swimming-pool': 0.0006887052341597796, 'ground-track-field': 0.09090909090909091, 'small-vehicle': 0.0037105751391465673, 'harbor': 0.002066115702479339, 'mAP': 0.04088832497272043, 'helicopter': 0.0, 'plane': 0.05622671901741669, 'large-vehicle': 0.0020815421512285627, 'ship': 0.007215007215007215, 'bridge': 0.0, 'soccer-ball-field': 0.022727272727272728, 'tennis-court': 0.3190853901779479, 'storage-tank': 0.06398751463129146, 'baseball-diamond': 0.022727272727272728, 'roundabout': 0.008912655971479501},
'0.5': {'basketball-court': 0.6294142968602868, 'swimming-pool': 0.5160393822507197, 'ground-track-field': 0.5687192477481807, 'small-vehicle': 0.601138143408285, 'harbor': 0.5469610986392011, 'mAP': 0.6500305186931586, 'helicopter': 0.43969264695357574, 'plane': 0.8971060614316362, 'large-vehicle': 0.6016309266962633, 'ship': 0.8075235131836705, 'bridge': 0.38219755010092094, 'soccer-ball-field': 0.698494842215314, 'tennis-court': 0.9049758181816826, 'storage-tank': 0.8176098661693649, 'baseball-diamond': 0.6873002469302452, 'roundabout': 0.6516541396280324},
'0.6': {'basketball-court': 0.6070779682338426, 'swimming-pool': 0.4087697000677784, 'ground-track-field': 0.5136450143567581, 'small-vehicle': 0.5434726849607782, 'harbor': 0.39132632505242537, 'mAP': 0.5828261571406362, 'helicopter': 0.3660366931918656, 'plane': 0.8927473766710153, 'large-vehicle': 0.502351287302958, 'ship': 0.7307724717469471, 'bridge': 0.2576193261211846, 'soccer-ball-field': 0.6474306589368679, 'tennis-court': 0.9028324650235644, 'storage-tank': 0.7818892176992728, 'baseball-diamond': 0.6205201124728893, 'roundabout': 0.5759010552713976},
'0.7': {'basketball-court': 0.5774064997428283, 'swimming-pool': 0.20357638933963368, 'ground-track-field': 0.43077298955236015, 'small-vehicle': 0.3711799948414751, 'harbor': 0.17653072296895098, 'mAP': 0.44785730052030076, 'helicopter': 0.30476834235527844, 'plane': 0.7866253773971212, 'large-vehicle': 0.3124728729878776, 'ship': 0.5780118107911296, 'bridge': 0.12581168831168832, 'soccer-ball-field': 0.556274607515316, 'tennis-court': 0.8123591608483569, 'storage-tank': 0.6663341483926283, 'baseball-diamond': 0.4118020444812819, 'roundabout': 0.403932858278585},
'0.55': {'basketball-court': 0.6245739719465224, 'swimming-pool': 0.4904772694381375, 'ground-track-field': 0.5472322176189219, 'small-vehicle': 0.5852951744371193, 'harbor': 0.4949711899201065, 'mAP': 0.6192209675973472, 'helicopter': 0.3918715110357292, 'plane': 0.8961318656338153, 'large-vehicle': 0.5485691464731728, 'ship': 0.7475687678471874, 'bridge': 0.32378980328968693, 'soccer-ball-field': 0.684192138967816, 'tennis-court': 0.9045695253380635, 'storage-tank': 0.7883960041036164, 'baseball-diamond': 0.6539522250694111, 'roundabout': 0.6067237028409027},
'0.75': {'basketball-court': 0.4955676150792878, 'swimming-pool': 0.1354409400123686, 'ground-track-field': 0.3202127635732761, 'small-vehicle': 0.24202996447883676, 'harbor': 0.10089820359281437, 'mAP': 0.34314637066383064, 'helicopter': 0.1430879038317055, 'plane': 0.6739837944462272, 'large-vehicle': 0.21417684138783752, 'ship': 0.37744353737870645, 'bridge': 0.04350016795431643, 'soccer-ball-field': 0.483003108003108, 'tennis-court': 0.8033977753129062, 'storage-tank': 0.548138300046708, 'baseball-diamond': 0.2703665635122973, 'roundabout': 0.2959480813470627}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20200918'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 2000
SAVE_WEIGHTS_INTE = 20673 * 2

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
ANGLE_WEIGHT = 0.5
REG_LOSS_MODE = None
ALPHA = 1.0
BETA = 1.0

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'DOTATrain'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 800
CLASS_NUM = 15
OMEGA = 180 / 64.
ANGLE_MODE = 0

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
ANGLE_RANGE = 180  # 90 or 180

# --------------------------------------------RPN config
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4

NMS = True
NMS_IOU_THRESHOLD = 0.1
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4


