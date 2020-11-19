# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 8.

{'0.7': {'tennis-court': 0.9030146006529745, 'ship': 0.5247147197113373, 'basketball-court': 0.5197288166873728, 'small-vehicle': 0.3876947548806594, 'harbor': 0.11656494591608674, 'plane': 0.7970783470996496, 'soccer-ball-field': 0.5481252919561383, 'ground-track-field': 0.49822883370115856, 'roundabout': 0.4251287737133813, 'baseball-diamond': 0.4246128098451668, 'large-vehicle': 0.2103852912661382, 'helicopter': 0.26122090469916553, 'bridge': 0.09114583333333333, 'swimming-pool': 0.14757313945494202, 'storage-tank': 0.6677333132875084, 'mAP': 0.4348633584136675},
'0.9': {'tennis-court': 0.19764274620067623, 'ship': 0.004702194357366771, 'basketball-court': 0.09090909090909091, 'small-vehicle': 0.0303030303030303, 'harbor': 0.0016175994823681658, 'plane': 0.09577370534335426, 'soccer-ball-field': 0.045454545454545456, 'ground-track-field': 0.004784688995215311, 'roundabout': 0.09090909090909091, 'baseball-diamond': 0.09090909090909091, 'large-vehicle': 0.0036363636363636364, 'helicopter': 0.0, 'bridge': 0.0, 'swimming-pool': 0.001652892561983471, 'storage-tank': 0.047402781720124895, 'mAP': 0.04704652138548674},
'0.85': {'tennis-court': 0.5175145387959259, 'ship': 0.04181083824704637, 'basketball-court': 0.16507177033492823, 'small-vehicle': 0.0606060606060606, 'harbor': 0.004132231404958678, 'plane': 0.31478501464749403, 'soccer-ball-field': 0.19507575757575757, 'ground-track-field': 0.007974481658692184, 'roundabout': 0.12648221343873517, 'baseball-diamond': 0.10730253353204174, 'large-vehicle': 0.005236915550816896, 'helicopter': 0.0606060606060606, 'bridge': 0.0303030303030303, 'swimming-pool': 0.003305785123966942, 'storage-tank': 0.2216713262889979, 'mAP': 0.1241252372076342},
'0.95': {'tennis-court': 0.011019283746556474, 'ship': 0.0034965034965034965, 'basketball-court': 0.0, 'small-vehicle': 0.00033921302578018993, 'harbor': 0.0, 'plane': 0.0303030303030303, 'soccer-ball-field': 0.004329004329004329, 'ground-track-field': 0.0, 'roundabout': 0.0101010101010101, 'baseball-diamond': 0.0, 'large-vehicle': 0.00016528925619834712, 'helicopter': 0.0, 'bridge': 0.0, 'swimming-pool': 0.0, 'storage-tank': 0.004914004914004914, 'mAP': 0.004311155944805877},
'0.75': {'tennis-court': 0.814202005991537, 'ship': 0.36943182805314534, 'basketball-court': 0.45146913919982956, 'small-vehicle': 0.2500155419128262, 'harbor': 0.04276380829572319, 'plane': 0.7579878981894648, 'soccer-ball-field': 0.4295376606872696, 'ground-track-field': 0.38142101120570016, 'roundabout': 0.33333075942849677, 'baseball-diamond': 0.32189281750059, 'large-vehicle': 0.08109584612393884, 'helicopter': 0.10013175230566534, 'bridge': 0.03636363636363637, 'swimming-pool': 0.04511019283746556, 'storage-tank': 0.5556670810786699, 'mAP': 0.3313613986115972},
'0.6': {'tennis-court': 0.9078675692919017, 'ship': 0.7428748965130202, 'basketball-court': 0.5525816701862102, 'small-vehicle': 0.5661458809978344, 'harbor': 0.31468024317286736, 'plane': 0.8927954483248337, 'soccer-ball-field': 0.7026712063326276, 'ground-track-field': 0.5952492478039144, 'roundabout': 0.5862217256587403, 'baseball-diamond': 0.6337310374828784, 'large-vehicle': 0.473337107335067, 'helicopter': 0.4559992150034522, 'bridge': 0.24367445113583264, 'swimming-pool': 0.3895201094294005, 'storage-tank': 0.780240020845799, 'mAP': 0.5891726553009586},
'0.65': {'tennis-court': 0.9063366415272575, 'ship': 0.6452205305336104, 'basketball-court': 0.5419974943230758, 'small-vehicle': 0.5169961021709822, 'harbor': 0.22418261562998404, 'plane': 0.8852270920046745, 'soccer-ball-field': 0.6228637775932758, 'ground-track-field': 0.5591851331213543, 'roundabout': 0.5511841761637736, 'baseball-diamond': 0.580333891442914, 'large-vehicle': 0.3714621290611434, 'helicopter': 0.38442656608097786, 'bridge': 0.17609532766667257, 'swimming-pool': 0.2673287170682332, 'storage-tank': 0.7642816542612352, 'mAP': 0.5331414565766109},
'0.5': {'tennis-court': 0.9088195386702851, 'ship': 0.8224437807168951, 'basketball-court': 0.5830775602074171, 'small-vehicle': 0.6169954809326167, 'harbor': 0.5258339843237152, 'plane': 0.8967687126422501, 'soccer-ball-field': 0.7362705406914213, 'ground-track-field': 0.6498421987512867, 'roundabout': 0.6566326127028347, 'baseball-diamond': 0.6993401680187941, 'large-vehicle': 0.6045608802509415, 'helicopter': 0.5212808419504471, 'bridge': 0.36652945756438354, 'swimming-pool': 0.5164216645404407, 'storage-tank': 0.820030826549724, 'mAP': 0.6616565499008968},
'0.8': {'tennis-court': 0.7816894723179306, 'ship': 0.1451783316171541, 'basketball-court': 0.3190681777298075, 'small-vehicle': 0.10194653796304984, 'harbor': 0.013468013468013467, 'plane': 0.5427055255026874, 'soccer-ball-field': 0.3415451418500199, 'ground-track-field': 0.13296378418329638, 'roundabout': 0.24207752583038625, 'baseball-diamond': 0.15874047455208096, 'large-vehicle': 0.02730096965512913, 'helicopter': 0.07382920110192837, 'bridge': 0.0303030303030303, 'swimming-pool': 0.022727272727272728, 'storage-tank': 0.3935350148663551, 'mAP': 0.22180523157787616},
'0.55': {'tennis-court': 0.9088195386702851, 'ship': 0.7579876783545256, 'basketball-court': 0.5756532396263904, 'small-vehicle': 0.6038512968643537, 'harbor': 0.41351532033351385, 'plane': 0.8953579426876774, 'soccer-ball-field': 0.7172833446523743, 'ground-track-field': 0.6291038290639339, 'roundabout': 0.6120712939504148, 'baseball-diamond': 0.6832929342136569, 'large-vehicle': 0.540009629017207, 'helicopter': 0.5193070813559969, 'bridge': 0.31825260872940675, 'swimming-pool': 0.4790914021824345, 'storage-tank': 0.78480241358501, 'mAP': 0.6292266368858123},
'mmAP': 0.3576710201805347}

"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20200920'
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
OMEGA = 180 / 8.
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


