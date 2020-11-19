# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 4.

{'0.75': {'helicopter': 0.10876623376623376, 'plane': 0.6261590820350442, 'storage-tank': 0.5697743174702269, 'tennis-court': 0.6026590467841741, 'swimming-pool': 0.040106951871657755, 'bridge': 0.0606060606060606, 'soccer-ball-field': 0.29083413775222616, 'harbor': 0.028655598899814838, 'roundabout': 0.33217013884400926, 'small-vehicle': 0.12618998165181367, 'baseball-diamond': 0.22509909218769975, 'ship': 0.10926192442325014, 'ground-track-field': 0.1988529807689143, 'mAP': 0.24875341013031635, 'basketball-court': 0.34781582054309323, 'large-vehicle': 0.06434978435052693},
'0.55': {'helicopter': 0.4744347006621738, 'plane': 0.8957642309025142, 'storage-tank': 0.7849056299207573, 'tennis-court': 0.9046144991020904, 'swimming-pool': 0.4731508326847427, 'bridge': 0.24568597431068456, 'soccer-ball-field': 0.6514604520734688, 'harbor': 0.3024824698269053, 'roundabout': 0.6358730574048375, 'small-vehicle': 0.5728596038534517, 'baseball-diamond': 0.6488637457385472, 'ship': 0.5813158626785325, 'ground-track-field': 0.596124558196834, 'mAP': 0.5835838887166079, 'basketball-court': 0.5873186403325683, 'large-vehicle': 0.39890407306100956},
'0.9': {'helicopter': 0.0, 'plane': 0.11460877431026685, 'storage-tank': 0.05750737274009488, 'tennis-court': 0.15913486899947246, 'swimming-pool': 0.0005107252298263534, 'bridge': 0.0, 'soccer-ball-field': 0.0606060606060606, 'harbor': 0.0004318721658389117, 'roundabout': 0.045454545454545456, 'small-vehicle': 0.003896103896103896, 'baseball-diamond': 0.045454545454545456, 'ship': 0.018181818181818184, 'ground-track-field': 0.0025974025974025974, 'mAP': 0.035686426500772424, 'basketball-court': 0.025974025974025972, 'large-vehicle': 0.000938281901584654},
'0.8': {'helicopter': 0.08181818181818182, 'plane': 0.404965437561218, 'storage-tank': 0.4232554615094123, 'tennis-court': 0.49650701718239065, 'swimming-pool': 0.012987012987012986, 'bridge': 0.018181818181818184, 'soccer-ball-field': 0.20363997651628055, 'harbor': 0.008621630470369967, 'roundabout': 0.23755411255411257, 'small-vehicle': 0.03881685575364668, 'baseball-diamond': 0.11923583662714096, 'ship': 0.06079248864483764, 'ground-track-field': 0.11931818181818182, 'mAP': 0.16699553570063064, 'basketball-court': 0.25691587823940765, 'large-vehicle': 0.02232314564544774},
'mmAP': 0.31009260103172187,
'0.7': {'helicopter': 0.21655188246097337, 'plane': 0.7837033364268121, 'storage-tank': 0.6766325937707176, 'tennis-court': 0.7531337747437782, 'swimming-pool': 0.09962947624794305, 'bridge': 0.09666622037756059, 'soccer-ball-field': 0.45707870730439126, 'harbor': 0.053616761464057326, 'roundabout': 0.4640797232366124, 'small-vehicle': 0.2537630457539992, 'baseball-diamond': 0.4081837275217716, 'ship': 0.16469794133073, 'ground-track-field': 0.32010188669067174, 'mAP': 0.3526239900570703, 'basketball-court': 0.41375480404895904, 'large-vehicle': 0.1277659694770764},
'0.65': {'helicopter': 0.3536633867607319, 'plane': 0.8835043774088882, 'storage-tank': 0.7637731652459822, 'tennis-court': 0.9030454652902554, 'swimming-pool': 0.25313548287637216, 'bridge': 0.11907810499359794, 'soccer-ball-field': 0.5691651795869328, 'harbor': 0.11986035247388582, 'roundabout': 0.5211534232745603, 'small-vehicle': 0.40825064051775517, 'baseball-diamond': 0.5417445700267612, 'ship': 0.2511909509245631, 'ground-track-field': 0.4690010735364718, 'mAP': 0.46115890049630265, 'basketball-court': 0.5510538855220554, 'large-vehicle': 0.20976344900572677},
'0.95': {'helicopter': 0.0, 'plane': 0.09090909090909091, 'storage-tank': 0.0053475935828877, 'tennis-court': 0.0303030303030303, 'swimming-pool': 0.0, 'bridge': 0.0, 'soccer-ball-field': 0.008264462809917356, 'harbor': 0.0, 'roundabout': 0.0, 'small-vehicle': 7.867511112859446e-05, 'baseball-diamond': 0.0, 'ship': 0.00028498147620404675, 'ground-track-field': 0.0, 'mAP': 0.009570436703119784, 'basketball-court': 0.008264462809917356, 'large-vehicle': 0.00010425354462051711},
'0.6': {'helicopter': 0.44018350039740417, 'plane': 0.8929091417760971, 'storage-tank': 0.7795868875383778, 'tennis-court': 0.9046144991020904, 'swimming-pool': 0.36610502311359944, 'bridge': 0.19395421735639892, 'soccer-ball-field': 0.6047875952081441, 'harbor': 0.2072308643466718, 'roundabout': 0.5992163694373649, 'small-vehicle': 0.5254517203883775, 'baseball-diamond': 0.6031967704339226, 'ship': 0.44168316330138013, 'ground-track-field': 0.5522957518738691, 'mAP': 0.5325808460769651, 'basketball-court': 0.5767922084787784, 'large-vehicle': 0.30070497840199967},
'0.85': {'helicopter': 0.013986013986013986, 'plane': 0.2245208016504695, 'storage-tank': 0.2164994910939014, 'tennis-court': 0.36938527311764985, 'swimming-pool': 0.012987012987012986, 'bridge': 0.0, 'soccer-ball-field': 0.09860509860509861, 'harbor': 0.0031451250561629475, 'roundabout': 0.09420289855072464, 'small-vehicle': 0.022727272727272728, 'baseball-diamond': 0.045454545454545456, 'ship': 0.045454545454545456, 'ground-track-field': 0.012987012987012986, 'mAP': 0.08614401812853563, 'basketball-court': 0.1278409090909091, 'large-vehicle': 0.0043642711667151585},
'0.5': {'helicopter': 0.48498648287488183, 'plane': 0.8981158855171345, 'storage-tank': 0.7871516406647231, 'tennis-court': 0.9051918689811524, 'swimming-pool': 0.523559435802154, 'bridge': 0.30744775808322455, 'soccer-ball-field': 0.6703088675833423, 'harbor': 0.40542090785094587, 'roundabout': 0.667214208043851, 'small-vehicle': 0.5953549814024383, 'baseball-diamond': 0.6751587500651139, 'ship': 0.7207120062792685, 'ground-track-field': 0.606089076120964, 'mAP': 0.6238285578068979, 'basketball-court': 0.6009269105967685, 'large-vehicle': 0.5097895872375081}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20200921'
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
OMEGA = 180 / 4.
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


