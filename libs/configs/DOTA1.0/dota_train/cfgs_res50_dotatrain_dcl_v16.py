# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 180.

{'0.65': {'soccer-ball-field': 0.5548311497496574, 'tennis-court': 0.9036699095809253, 'baseball-diamond': 0.5794656068293492, 'mAP': 0.5441584018713088, 'bridge': 0.20131184515827558, 'storage-tank': 0.7427602540936564, 'roundabout': 0.5309099225681633, 'ship': 0.6462835661450131, 'plane': 0.8824495829319922, 'swimming-pool': 0.34596207644621074, 'helicopter': 0.4126652027238286, 'large-vehicle': 0.43855995093349637, 'basketball-court': 0.567636800097396, 'small-vehicle': 0.5154890566594604, 'ground-track-field': 0.5530433147635094, 'harbor': 0.2873377893886979},
'0.5': {'soccer-ball-field': 0.6928701286061683, 'tennis-court': 0.9052715331826937, 'baseball-diamond': 0.6933101873137151, 'mAP': 0.6583099499467682, 'bridge': 0.37416095488305356, 'storage-tank': 0.8121702724857688, 'roundabout': 0.65512005910603, 'ship': 0.807343411150596, 'plane': 0.8984451668669173, 'swimming-pool': 0.5366958996008825, 'helicopter': 0.511213143818906, 'large-vehicle': 0.5869886053199255, 'basketball-court': 0.6076214636916593, 'small-vehicle': 0.6331323087943526, 'ground-track-field': 0.6231062283240522, 'harbor': 0.5371998860567999},
'0.9': {'soccer-ball-field': 0.09155844155844155, 'tennis-court': 0.3018839371422833, 'baseball-diamond': 0.009569377990430622, 'mAP': 0.04789656534165517, 'bridge': 0.0, 'storage-tank': 0.02323232323232323, 'roundabout': 0.045454545454545456, 'ship': 0.005145797598627788, 'plane': 0.11069733479372033, 'swimming-pool': 0.000368052999631947, 'helicopter': 0.0, 'large-vehicle': 0.0015477762097691992, 'basketball-court': 0.025974025974025972, 'small-vehicle': 0.011363636363636364, 'ground-track-field': 0.09090909090909091, 'harbor': 0.0007441398983008805},
'0.8': {'soccer-ball-field': 0.2771359075706902, 'tennis-court': 0.7848120465093688, 'baseball-diamond': 0.1476640814766408, 'mAP': 0.23438562375972374, 'bridge': 0.09090909090909091, 'storage-tank': 0.3329777637614456, 'roundabout': 0.2551868802440885, 'ship': 0.21213549592187783, 'plane': 0.5017225007152174, 'swimming-pool': 0.0606060606060606, 'helicopter': 0.07707509881422925, 'large-vehicle': 0.06292037377880987, 'basketball-court': 0.3640787069036866, 'small-vehicle': 0.07776624210595809, 'ground-track-field': 0.2253395616241466, 'harbor': 0.045454545454545456},
'0.6': {'soccer-ball-field': 0.6178126145403033, 'tennis-court': 0.9047256108742666, 'baseball-diamond': 0.6473266901133993, 'mAP': 0.5977967756274896, 'bridge': 0.2583938624693473, 'storage-tank': 0.7741734035312222, 'roundabout': 0.5789441782782515, 'ship': 0.739788646463669, 'plane': 0.8939874951511768, 'swimming-pool': 0.43992970601463793, 'helicopter': 0.464918863069194, 'large-vehicle': 0.5116895944803548, 'basketball-court': 0.5921592377455623, 'small-vehicle': 0.5711791088686868, 'ground-track-field': 0.5859187298644489, 'harbor': 0.3860038929478212},
'0.85': {'soccer-ball-field': 0.21801727684080624, 'tennis-court': 0.6293525766353285, 'baseball-diamond': 0.018181818181818184, 'mAP': 0.12474199550269245, 'bridge': 0.09090909090909091, 'storage-tank': 0.16586845926738025, 'roundabout': 0.07423985182898596, 'ship': 0.03840003533881085, 'plane': 0.2796453026683376, 'swimming-pool': 0.0303030303030303, 'helicopter': 0.013468013468013467, 'large-vehicle': 0.009852216748768473, 'basketball-court': 0.15016876002877222, 'small-vehicle': 0.0303030303030303, 'ground-track-field': 0.11492281303602059, 'harbor': 0.007497656982193065},
'0.75': {'soccer-ball-field': 0.3881573881573882, 'tennis-court': 0.7989419715747543, 'baseball-diamond': 0.21840354767184036, 'mAP': 0.33943646598247734, 'bridge': 0.09905303030303031, 'storage-tank': 0.534407819864381, 'roundabout': 0.381678796448254, 'ship': 0.3957926493612071, 'plane': 0.660214075693295, 'swimming-pool': 0.10471763085399449, 'helicopter': 0.2080808080808081, 'large-vehicle': 0.15400546515925337, 'basketball-court': 0.48735654655565663, 'small-vehicle': 0.21838315849243103, 'ground-track-field': 0.36779756050090984, 'harbor': 0.07455654101995565},
'0.7': {'soccer-ball-field': 0.4689738835194373, 'tennis-court': 0.8871659629246796, 'baseball-diamond': 0.4025165121465333, 'mAP': 0.4529081527723654, 'bridge': 0.1310001022599448, 'storage-tank': 0.6567730533210858, 'roundabout': 0.4727243127533465, 'ship': 0.5766909116194356, 'plane': 0.7870953915970476, 'swimming-pool': 0.17938650848159987, 'helicopter': 0.3499287058005919, 'large-vehicle': 0.34118400637241086, 'basketball-court': 0.5302570987921067, 'small-vehicle': 0.37089044636021734, 'ground-track-field': 0.4954125912747115, 'harbor': 0.14362280436233354},
'0.55': {'soccer-ball-field': 0.6447099331112448, 'tennis-court': 0.9052715331826937, 'baseball-diamond': 0.6668596608517259, 'mAP': 0.6312320711702848, 'bridge': 0.3434812701166685, 'storage-tank': 0.7841732022238794, 'roundabout': 0.6447304604175382, 'ship': 0.7550728666524944, 'plane': 0.8966102057790738, 'swimming-pool': 0.4821034726145486, 'helicopter': 0.48297759107372545, 'large-vehicle': 0.5602866866422265, 'basketball-court': 0.6049897210452538, 'small-vehicle': 0.6175960136059865, 'ground-track-field': 0.6063957055485166, 'harbor': 0.47322274468869663},
'mmAP': 0.3635483348570584,
'0.95': {'soccer-ball-field': 0.012987012987012986, 'tennis-court': 0.025974025974025972, 'baseball-diamond': 0.0, 'mAP': 0.0046173465958186865, 'bridge': 0.0, 'storage-tank': 0.004132231404958678, 'roundabout': 0.022727272727272728, 'ship': 3.5025656293234796e-05, 'plane': 0.003134796238244514, 'swimming-pool': 0.0, 'helicopter': 0.0, 'large-vehicle': 0.0, 'basketball-court': 0.0, 'small-vehicle': 0.0001164008846467233, 'ground-track-field': 0.0, 'harbor': 0.00015343306482546988}}

"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20200929'
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
OMEGA = 180 / 180.
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


