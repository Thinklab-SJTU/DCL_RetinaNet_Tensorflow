# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
GCL + OMEGA = 180 / 256.

{'0.55': {'large-vehicle': 0.5363181598334215, 'bridge': 0.32699022291214586, 'roundabout': 0.6834084685791627, 'helicopter': 0.42423986472193886, 'basketball-court': 0.6356572293817766, 'soccer-ball-field': 0.6356666551126983, 'plane': 0.8945924414158641, 'swimming-pool': 0.4770219697624021, 'ship': 0.7496911805782385, 'ground-track-field': 0.5263486334029331, 'harbor': 0.4815265990046088, 'small-vehicle': 0.5829936682145498, 'mAP': 0.6212560940651672, 'tennis-court': 0.9073456479981408, 'baseball-diamond': 0.6691949779767582, 'storage-tank': 0.787845692082869},
'0.65': {'large-vehicle': 0.4165010382816529, 'bridge': 0.20701815296933468, 'roundabout': 0.5891087444485502, 'helicopter': 0.2460068666965219, 'basketball-court': 0.615703648566293, 'soccer-ball-field': 0.5995826390342202, 'plane': 0.8789449288346928, 'swimming-pool': 0.30728237011916837, 'ship': 0.7006510845615044, 'ground-track-field': 0.4773757093225361, 'harbor': 0.27759170853965937, 'small-vehicle': 0.5004502999772011, 'mAP': 0.5336012253397696, 'tennis-court': 0.9065210815577387, 'baseball-diamond': 0.5171066882224982, 'storage-tank': 0.7641734189649739},
'0.7': {'large-vehicle': 0.322000445790306, 'bridge': 0.15630595927206098, 'roundabout': 0.45747372020663946, 'helicopter': 0.1581344490658216, 'basketball-court': 0.5836723390158411, 'soccer-ball-field': 0.5113428986870161, 'plane': 0.7821462959244937, 'swimming-pool': 0.15549132248590805, 'ship': 0.5990704063887706, 'ground-track-field': 0.4232355483112566, 'harbor': 0.15442246589529374, 'small-vehicle': 0.37595739362963715, 'mAP': 0.4423366550900184, 'tennis-court': 0.8993422827314298, 'baseball-diamond': 0.38789177156656274, 'storage-tank': 0.668562527379238},
'0.8': {'large-vehicle': 0.08634972958890115, 'bridge': 0.09090909090909091, 'roundabout': 0.2598233191016696, 'helicopter': 0.06511056511056511, 'basketball-court': 0.35725607521363945, 'soccer-ball-field': 0.3345961412852878, 'plane': 0.5085858846538271, 'swimming-pool': 0.00809464508094645, 'ship': 0.22290872579083695, 'ground-track-field': 0.24661305953959506, 'harbor': 0.045454545454545456, 'small-vehicle': 0.08799270904534062, 'mAP': 0.2419578004776401, 'tennis-court': 0.7925837550839652, 'baseball-diamond': 0.17168809808612442, 'storage-tank': 0.351400663220267},
'0.95': {'large-vehicle': 6.216714673518867e-05, 'bridge': 0.0, 'roundabout': 0.01515151515151515, 'helicopter': 0.0, 'basketball-court': 0.0, 'soccer-ball-field': 0.003367003367003367, 'plane': 0.0010822510822510823, 'swimming-pool': 0.0, 'ship': 0.003367003367003367, 'ground-track-field': 0.00048100048100048096, 'harbor': 3.4897923573547374e-05, 'small-vehicle': 0.00017024174327545115, 'mAP': 0.005180805617224108, 'tennis-court': 0.045454545454545456, 'baseball-diamond': 0.004545454545454546, 'storage-tank': 0.003996003996003996},
'0.6': {'large-vehicle': 0.49043148824212784, 'bridge': 0.2665501920436261, 'roundabout': 0.6382530988858008, 'helicopter': 0.3348544500119303, 'basketball-court': 0.6356572293817766, 'soccer-ball-field': 0.6209492441260729, 'plane': 0.8914063938141928, 'swimming-pool': 0.37634044247127274, 'ship': 0.7368647800084778, 'ground-track-field': 0.5031943314393181, 'harbor': 0.3789057919679977, 'small-vehicle': 0.5435040539045011, 'mAP': 0.5824613319415164, 'tennis-court': 0.9068000669906829, 'baseball-diamond': 0.6311591796320877, 'storage-tank': 0.7820492362028806},
'0.85': {'large-vehicle': 0.02296657337264188, 'bridge': 0.008912655971479501, 'roundabout': 0.139213270920588, 'helicopter': 0.002457002457002457, 'basketball-court': 0.12763335182690022, 'soccer-ball-field': 0.19858665708468082, 'plane': 0.297514597175563, 'swimming-pool': 0.0019527012367107832, 'ship': 0.11203802733214499, 'ground-track-field': 0.15469554030874785, 'harbor': 0.0303030303030303, 'small-vehicle': 0.025454545454545455, 'mAP': 0.13613529778752925, 'tennis-court': 0.673010338947504, 'baseball-diamond': 0.06254272043745727, 'storage-tank': 0.18474845398394302},
'mmAP': 0.35970197506551804, '0.75': {'large-vehicle': 0.2135917899475711, 'bridge': 0.1162534435261708, 'roundabout': 0.34408871050092427, 'helicopter': 0.12776412776412777, 'basketball-court': 0.48116837373192944, 'soccer-ball-field': 0.4438988013792348, 'plane': 0.6626048644825645, 'swimming-pool': 0.036884708658320775, 'ship': 0.4044876846482558, 'ground-track-field': 0.31903870347806274, 'harbor': 0.08016848016848016, 'small-vehicle': 0.23003516067617347, 'mAP': 0.3377308387743835, 'tennis-court': 0.8102634625783773, 'baseball-diamond': 0.2504702630857191, 'storage-tank': 0.5452440069898383},
'0.5': {'large-vehicle': 0.5835269354138677, 'bridge': 0.35807744831931165, 'roundabout': 0.687317274273796, 'helicopter': 0.48276792710930594, 'basketball-court': 0.6356572293817766, 'soccer-ball-field': 0.6479814539632792, 'plane': 0.8967862211777952, 'swimming-pool': 0.5194988614123708, 'ship': 0.8144620817830355, 'ground-track-field': 0.551501641561227, 'harbor': 0.5378286729265327, 'small-vehicle': 0.5983602806406438, 'mAP': 0.6486567230293808, 'tennis-court': 0.9078693742155282, 'baseball-diamond': 0.6917829240349932, 'storage-tank': 0.816432519227246},
'0.9': {'large-vehicle': 0.0023715415019762843, 'bridge': 0.0002154243860404998, 'roundabout': 0.025974025974025972, 'helicopter': 0.002457002457002457, 'basketball-court': 0.013636363636363636, 'soccer-ball-field': 0.06433566433566434, 'plane': 0.05181262284341559, 'swimming-pool': 0.0007951232441028359, 'ship': 0.01652892561983471, 'ground-track-field': 0.022727272727272728, 'harbor': 0.012987012987012986, 'small-vehicle': 0.0036363636363636364, 'mAP': 0.04770297853255096, 'tennis-court': 0.38293522377602146, 'baseball-diamond': 0.045454545454545456, 'storage-tank': 0.06967756540862197}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_G_2x_20200915'
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
OMEGA = 180 / 256.
ANGLE_MODE = 1

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


