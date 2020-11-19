# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
CSL + gaussian label, omega=1, r=6

{'0.55': {'helicopter': 0.4550492130641718, 'swimming-pool': 0.5028725968595296, 'roundabout': 0.625774455252499, 'baseball-diamond': 0.6016939772502411, 'storage-tank': 0.7820540650201766, 'mAP': 0.6198733244240139, 'ground-track-field': 0.5445535571306294, 'small-vehicle': 0.6072399880746298, 'bridge': 0.3348778821842333, 'soccer-ball-field': 0.6411316028963089, 'basketball-court': 0.6083651368714932, 'tennis-court': 0.9082761879222057, 'ship': 0.7607677865897701, 'large-vehicle': 0.5593603669665573, 'plane': 0.894590775112624, 'harbor': 0.471492275165142},
'mmAP': 0.35036738193711736,
'0.95': {'helicopter': 0.0, 'swimming-pool': 0.0, 'roundabout': 0.0, 'baseball-diamond': 0.0, 'storage-tank': 0.0006384065372829418, 'mAP': 0.0030430163948947098, 'ground-track-field': 0.0, 'small-vehicle': 6.952007971635808e-05, 'bridge': 0.0, 'soccer-ball-field': 0.005681818181818182, 'basketball-court': 0.022727272727272728, 'tennis-court': 0.012987012987012986, 'ship': 0.0007331378299120235, 'large-vehicle': 5.3256643766309845e-05, 'plane': 0.0027548209366391185, 'harbor': 0.0},
'0.7': {'helicopter': 0.24444849245008324, 'swimming-pool': 0.21717783370032506, 'roundabout': 0.4101043166176963, 'baseball-diamond': 0.3696307676693788, 'storage-tank': 0.5831698189830972, 'mAP': 0.43130987537016724, 'ground-track-field': 0.38180720383439154, 'small-vehicle': 0.34152486751905664, 'bridge': 0.15370329057445623, 'soccer-ball-field': 0.48581835095193693, 'basketball-court': 0.5510437326258193, 'tennis-court': 0.9030283547204782, 'ship': 0.5286344669343855, 'large-vehicle': 0.31518463906963523, 'plane': 0.7864238599055797, 'harbor': 0.19794813499618763},
'0.8': {'helicopter': 0.05252525252525253, 'swimming-pool': 0.039468252002311945, 'roundabout': 0.21636785080864662, 'baseball-diamond': 0.15495867768595042, 'storage-tank': 0.27054205461747594, 'mAP': 0.22584316957941405, 'ground-track-field': 0.17680940002320572, 'small-vehicle': 0.07554977739376523, 'bridge': 0.025, 'soccer-ball-field': 0.3351821604537089, 'basketball-court': 0.3694610026210497, 'tennis-court': 0.7725341096146024, 'ship': 0.20779293081827577, 'large-vehicle': 0.11168838177520103, 'plane': 0.4888586024426739, 'harbor': 0.09090909090909091},
'0.85': {'helicopter': 0.022727272727272728, 'swimming-pool': 0.0036363636363636364, 'roundabout': 0.07210031347962383, 'baseball-diamond': 0.05, 'storage-tank': 0.09852715590420508, 'mAP': 0.1134067490217653, 'ground-track-field': 0.07808857808857808, 'small-vehicle': 0.01073322125953705, 'bridge': 0.005681818181818182, 'soccer-ball-field': 0.25354683195592287, 'basketball-court': 0.14976689976689975, 'tennis-court': 0.5564283405487576, 'ship': 0.1069391291790021, 'large-vehicle': 0.06887751830621064, 'plane': 0.21150860733930982, 'harbor': 0.012539184952978056},
'0.6': {'helicopter': 0.36616774952166214, 'swimming-pool': 0.42280468169415375, 'roundabout': 0.5515578887546573, 'baseball-diamond': 0.5566001286701034, 'storage-tank': 0.752862937026529, 'mAP': 0.5780082584800036, 'ground-track-field': 0.538648402990965, 'small-vehicle': 0.559942823754123, 'bridge': 0.26515120963702526, 'soccer-ball-field': 0.6217728067920131, 'basketball-court': 0.5949182030453426, 'tennis-court': 0.9082761879222057, 'ship': 0.7441728443470911, 'large-vehicle': 0.5105296730979295, 'plane': 0.8897962157325949, 'harbor': 0.38692212421365885},
'0.5': {'helicopter': 0.47411054854055434, 'swimming-pool': 0.5353678434935341, 'roundabout': 0.6540774239694102, 'baseball-diamond': 0.6409215533625178, 'storage-tank': 0.7887780293007403, 'mAP': 0.6440052304816558, 'ground-track-field': 0.5569179013668764, 'small-vehicle': 0.621375466867625, 'bridge': 0.35939856441323564, 'soccer-ball-field': 0.6537498444352045, 'basketball-court': 0.6083651368714932, 'tennis-court': 0.9082761879222057, 'ship': 0.8180999806409668, 'large-vehicle': 0.610588309703525, 'plane': 0.8958243800572547, 'harbor': 0.5342272862796947},
'0.65': {'helicopter': 0.30336400695510357, 'swimming-pool': 0.33147458694730025, 'roundabout': 0.47864032095686837, 'baseball-diamond': 0.4912136166895381, 'storage-tank': 0.6730405311495307, 'mAP': 0.5151195467796492, 'ground-track-field': 0.5115591993339109, 'small-vehicle': 0.4886852318765086, 'bridge': 0.20318870297523894, 'soccer-ball-field': 0.5705379588511073, 'basketball-court': 0.5715874819730236, 'tennis-court': 0.9056225638480788, 'ship': 0.6451715219709926, 'large-vehicle': 0.4094424853985508, 'plane': 0.8526646752622546, 'harbor': 0.29060031750673065},
'0.75': {'helicopter': 0.16666666666666666, 'swimming-pool': 0.09366068396440332, 'roundabout': 0.31746740320064804, 'baseball-diamond': 0.24085109511759967, 'storage-tank': 0.4270589684794223, 'mAP': 0.32584993285698344, 'ground-track-field': 0.2625838423763325, 'small-vehicle': 0.19982561995109147, 'bridge': 0.06117178276269185, 'soccer-ball-field': 0.43978576400483094, 'basketball-court': 0.48402304126942297, 'tennis-court': 0.8103850287719002, 'ship': 0.3843262612479747, 'large-vehicle': 0.21524407027394005, 'plane': 0.6682713325509215, 'harbor': 0.11642743221690591},
'0.9': {'helicopter': 0.00202020202020202, 'swimming-pool': 0.0, 'roundabout': 0.045454545454545456, 'baseball-diamond': 0.0303030303030303, 'storage-tank': 0.045454545454545456, 'mAP': 0.04721471598262659, 'ground-track-field': 0.013368983957219251, 'small-vehicle': 0.00267379679144385, 'bridge': 0.005681818181818182, 'soccer-ball-field': 0.07828282828282829, 'basketball-court': 0.03636363636363637, 'tennis-court': 0.30478925935313295, 'ship': 0.09090909090909091, 'large-vehicle': 0.00505050505050505, 'plane': 0.04613839381117237, 'harbor': 0.0017301038062283738}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_CSL_2x_20200916'
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
LABEL_TYPE = 0
RADUIUS = 6
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

