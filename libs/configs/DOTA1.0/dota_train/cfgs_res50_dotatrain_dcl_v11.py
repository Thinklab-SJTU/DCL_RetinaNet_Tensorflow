# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
GCL + OMEGA = 180 / 32.
FLOPs: 867420999;    Trainable params: 33196536
{'0.8': {'tennis-court': 0.7881468448604563, 'ground-track-field': 0.20156279127570995, 'baseball-diamond': 0.1618181818181818, 'soccer-ball-field': 0.3890338110165697, 'ship': 0.17055881908005666, 'small-vehicle': 0.1206962821695297, 'swimming-pool': 0.010909090909090908, 'plane': 0.5312362002669713, 'bridge': 0.045454545454545456, 'harbor': 0.09090909090909091, 'basketball-court': 0.42867060897751436, 'storage-tank': 0.3265089926751668, 'large-vehicle': 0.08561994905820572, 'mAP': 0.24027857825942725, 'roundabout': 0.18252055005981707, 'helicopter': 0.07053291536050157},
'0.65': {'tennis-court': 0.9055554067423018, 'ground-track-field': 0.5238884115023222, 'baseball-diamond': 0.506390673258613, 'soccer-ball-field': 0.5763993246979054, 'ship': 0.708694768973092, 'small-vehicle': 0.49868950387167243, 'swimming-pool': 0.29791313513476825, 'plane': 0.8771289063758806, 'bridge': 0.1633223664739076, 'harbor': 0.3025352844354984, 'basketball-court': 0.5835593962872078, 'storage-tank': 0.7652771235218994, 'large-vehicle': 0.4380963220519667, 'mAP': 0.5341749190445875, 'roundabout': 0.5170986181015585, 'helicopter': 0.3480745442402185},
'0.9': {'tennis-court': 0.31090320713022956, 'ground-track-field': 0.0303030303030303, 'baseball-diamond': 0.0606060606060606, 'soccer-ball-field': 0.0718475073313783, 'ship': 0.005681818181818182, 'small-vehicle': 0.0024327784891165173, 'swimming-pool': 0.0010695187165775401, 'plane': 0.07372715779795426, 'bridge': 0.001567398119122257, 'harbor': 0.012987012987012986, 'basketball-court': 0.045454545454545456, 'storage-tank': 0.01948051948051948, 'large-vehicle': 0.004662004662004662, 'mAP': 0.04473503930416003, 'roundabout': 0.0303030303030303, 'helicopter': 0.0},
'0.85': {'tennis-court': 0.6668328432336217, 'ground-track-field': 0.11688311688311687, 'baseball-diamond': 0.07896270396270395, 'soccer-ball-field': 0.2743656343656343, 'ship': 0.04252324943909101, 'small-vehicle': 0.0303030303030303, 'swimming-pool': 0.003331746787244169, 'plane': 0.28500990595364734, 'bridge': 0.045454545454545456, 'harbor': 0.018181818181818184, 'basketball-court': 0.16896316233479292, 'storage-tank': 0.1300762263649527, 'large-vehicle': 0.024997131378868245, 'mAP': 0.1320718829305145, 'roundabout': 0.08382949295101398, 'helicopter': 0.011363636363636364},
'0.95': {'tennis-court': 0.0303030303030303, 'ground-track-field': 0.0, 'baseball-diamond': 0.004434589800443459, 'soccer-ball-field': 0.0, 'ship': 0.0012626262626262625, 'small-vehicle': 0.000505050505050505, 'swimming-pool': 0.0, 'plane': 0.002525252525252525, 'bridge': 0.0, 'harbor': 0.0, 'basketball-court': 0.0, 'storage-tank': 0.0030303030303030303, 'large-vehicle': 0.000157736999842263, 'mAP': 0.0028145726284365563, 'roundabout': 0.0, 'helicopter': 0.0},
'0.7': {'tennis-court': 0.9039716045171303, 'ground-track-field': 0.4308629435067456, 'baseball-diamond': 0.3267496730911365, 'soccer-ball-field': 0.5151867512332629, 'ship': 0.6005165685301741, 'small-vehicle': 0.3873208180174507, 'swimming-pool': 0.17817486561396792, 'plane': 0.7894831287499449, 'bridge': 0.10743524321925786, 'harbor': 0.21299716932719948, 'basketball-court': 0.546852427170258, 'storage-tank': 0.674142167828578, 'large-vehicle': 0.3293974227066024, 'mAP': 0.4467399176162477, 'roundabout': 0.4589751753866012, 'helicopter': 0.23903280534540436},
'0.6': {'tennis-court': 0.9068599294482994, 'ground-track-field': 0.5360679648689316, 'baseball-diamond': 0.6133620082765716, 'soccer-ball-field': 0.6292001293294663, 'ship': 0.7462008423879534, 'small-vehicle': 0.5522483089149173, 'swimming-pool': 0.3891941263842111, 'plane': 0.8916962169278684, 'bridge': 0.2592393425365398, 'harbor': 0.41188164823631396, 'basketball-court': 0.5933048156574457, 'storage-tank': 0.7807279875372064, 'large-vehicle': 0.513451393441771, 'mAP': 0.5890750841786159, 'roundabout': 0.6034291180937185, 'helicopter': 0.409262430638024},
'mmAP': 0.36151799964663417,
'0.55': {'tennis-court': 0.9073892803364623, 'ground-track-field': 0.5628313826043503, 'baseball-diamond': 0.6798761831924777, 'soccer-ball-field': 0.6730067200880602, 'ship': 0.7606462881164076, 'small-vehicle': 0.5932308221305281, 'swimming-pool': 0.4697437156106161, 'plane': 0.8942295148978198, 'bridge': 0.32470000168562396, 'harbor': 0.5099548736700419, 'basketball-court': 0.6220220164501689, 'storage-tank': 0.7871194369652578, 'large-vehicle': 0.5511423620375699, 'mAP': 0.6275703454947114, 'roundabout': 0.638697448850658, 'helicopter': 0.4389651357846286},
'0.5': {'tennis-court': 0.9084089506525566, 'ground-track-field': 0.5733829327818472, 'baseball-diamond': 0.6884249533725706, 'soccer-ball-field': 0.6758546303062845, 'ship': 0.8275237100463968, 'small-vehicle': 0.612427764173217, 'swimming-pool': 0.5149586506108559, 'plane': 0.8947809541140259, 'bridge': 0.3804404666369143, 'harbor': 0.5477265338688675, 'basketball-court': 0.6220220164501689, 'storage-tank': 0.7909595329078206, 'large-vehicle': 0.6087985575594734, 'mAP': 0.6510670034748898, 'roundabout': 0.6785853468778275, 'helicopter': 0.44171005176451994},
'0.75': {'tennis-court': 0.8058285556162824, 'ground-track-field': 0.3466423418230095, 'baseball-diamond': 0.24344165354762407, 'soccer-ball-field': 0.4738913218564382, 'ship': 0.4034487210946083, 'small-vehicle': 0.2419618607304998, 'swimming-pool': 0.050078194057251654, 'plane': 0.6674261484214561, 'bridge': 0.05566104913363399, 'harbor': 0.1365893748609877, 'basketball-court': 0.5426604637323884, 'storage-tank': 0.5440879066174735, 'large-vehicle': 0.21870448301103507, 'mAP': 0.3466526535347514, 'roundabout': 0.29361015276100666, 'helicopter': 0.17575757575757575}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_G_2x_20200925'
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
OMEGA = 180 / 32.
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


