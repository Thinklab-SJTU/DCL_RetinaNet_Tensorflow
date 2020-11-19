# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 32.

{'mmAP': 0.3670689860291625,
'0.75': {'roundabout': 0.34256254625795496, 'mAP': 0.35664632217558573, 'swimming-pool': 0.09320709524791158, 'storage-tank': 0.5484857110290995, 'plane': 0.6772875821974348, 'large-vehicle': 0.1931457833379231, 'helicopter': 0.19264990328820117, 'harbor': 0.14145996072608352, 'basketball-court': 0.47265894236482475, 'bridge': 0.09995791245791247, 'small-vehicle': 0.25495175035555634, 'ground-track-field': 0.35031220154669923, 'soccer-ball-field': 0.46871195576252267, 'baseball-diamond': 0.32482997558768756, 'ship': 0.38084189834492777, 'tennis-court': 0.808631614129047},
'0.6': {'roundabout': 0.5887751164227217, 'mAP': 0.5891947129981994, 'swimming-pool': 0.45552935536531064, 'storage-tank': 0.7830218428741531, 'plane': 0.8935015798612127, 'large-vehicle': 0.5147998562187823, 'helicopter': 0.3938536306460835, 'harbor': 0.4005285633385585, 'basketball-court': 0.5730678635090399, 'bridge': 0.26453915671670536, 'small-vehicle': 0.5450670123549419, 'ground-track-field': 0.5693905348563788, 'soccer-ball-field': 0.6083527468225252, 'baseball-diamond': 0.6138259657533816, 'ship': 0.7281807381796711, 'tennis-court': 0.905486732053524},
'0.85': {'roundabout': 0.11647727272727273, 'mAP': 0.14191292072133216, 'swimming-pool': 0.006269592476489028, 'storage-tank': 0.15647973821050742, 'plane': 0.31031435603880997, 'large-vehicle': 0.022727272727272728, 'helicopter': 0.045454545454545456, 'harbor': 0.09090909090909091, 'basketball-court': 0.11489459607143884, 'bridge': 0.010432190760059613, 'small-vehicle': 0.09090909090909091, 'ground-track-field': 0.13368983957219252, 'soccer-ball-field': 0.2095783004873914, 'baseball-diamond': 0.11989459815546771, 'ship': 0.03672152208491374, 'tennis-court': 0.6639418042354395},
'0.55': {'roundabout': 0.6422592294611053, 'mAP': 0.6278449763867877, 'swimming-pool': 0.498416979814446, 'storage-tank': 0.7882305849698752, 'plane': 0.897270784625041, 'large-vehicle': 0.555841590946543, 'helicopter': 0.45659445950681493, 'harbor': 0.49902932066860495, 'basketball-court': 0.5730678635090399, 'bridge': 0.3339537629008831, 'small-vehicle': 0.58581053496245, 'ground-track-field': 0.6015529157211015, 'soccer-ball-field': 0.6537448537691369, 'baseball-diamond': 0.6765755891738808, 'ship': 0.749317081784965, 'tennis-court': 0.906009093987929},
'0.65': {'roundabout': 0.5155757962346261, 'mAP': 0.5301516872544509, 'swimming-pool': 0.33092937001851286, 'storage-tank': 0.7614882533451384, 'plane': 0.8799136702793835, 'large-vehicle': 0.44093712843192245, 'helicopter': 0.3234402852049911, 'harbor': 0.29208833952341007, 'basketball-court': 0.55514343157982, 'bridge': 0.1776583569687018, 'small-vehicle': 0.49407742731054133, 'ground-track-field': 0.5367302058899696, 'soccer-ball-field': 0.574136485759996, 'baseball-diamond': 0.5360163253907396, 'ship': 0.6302171473145658, 'tennis-court': 0.9039230855644451},
'0.8': {'roundabout': 0.22735806121287974, 'mAP': 0.24779781650010155, 'swimming-pool': 0.057174532784288884, 'storage-tank': 0.35872829854284616, 'plane': 0.5244095577890446, 'large-vehicle': 0.0880023004062129, 'helicopter': 0.0888047138047138, 'harbor': 0.10060918462980319, 'basketball-court': 0.3753365175857577, 'bridge': 0.0303030303030303, 'small-vehicle': 0.14224876078424137, 'ground-track-field': 0.2362423628381075, 'soccer-ball-field': 0.3243589743589743, 'baseball-diamond': 0.21497323204245986, 'ship': 0.1507986198641653, 'tennis-court': 0.7976191005549975},
'0.95': {'roundabout': 0.0008576329331046312, 'mAP': 0.00471060036752231, 'swimming-pool': 0.0, 'storage-tank': 0.002190580503833516, 'plane': 0.002207505518763797, 'large-vehicle': 0.0002017213555675094, 'helicopter': 0.0, 'harbor': 0.0, 'basketball-court': 0.0, 'bridge': 0.0, 'small-vehicle': 0.003367003367003367, 'ground-track-field': 0.0, 'soccer-ball-field': 0.01515151515151515, 'baseball-diamond': 0.0, 'ship': 0.0012285012285012285, 'tennis-court': 0.045454545454545456},
'0.5': {'roundabout': 0.6947351393050408, 'mAP': 0.6592809274070299, 'swimming-pool': 0.5214646971428248, 'storage-tank': 0.8156719910833012, 'plane': 0.8990600505511154, 'large-vehicle': 0.6062214534580893, 'helicopter': 0.5035874208384727, 'harbor': 0.5440695691218229, 'basketball-court': 0.5819345573533122, 'bridge': 0.39834366548568256, 'small-vehicle': 0.6031838350085043, 'ground-track-field': 0.6335485831844249, 'soccer-ball-field': 0.6700987767612586, 'baseball-diamond': 0.7032315892192376, 'ship': 0.8080534886044332, 'tennis-court': 0.906009093987929},
'0.9': {'roundabout': 0.045454545454545456, 'mAP': 0.054111762885984474, 'swimming-pool': 0.0005078720162519046, 'storage-tank': 0.03636363636363637, 'plane': 0.11983471074380166, 'large-vehicle': 0.003952569169960474, 'helicopter': 0.0, 'harbor': 0.011363636363636364, 'basketball-court': 0.022727272727272728, 'bridge': 0.006734006734006734, 'small-vehicle': 0.006734006734006734, 'ground-track-field': 0.09090909090909091, 'soccer-ball-field': 0.045454545454545456, 'baseball-diamond': 0.09090909090909091, 'ship': 0.006993006993006993, 'tennis-court': 0.32373845271691454},
'0.7': {'roundabout': 0.459196129965961, 'mAP': 0.45903813359463097, 'swimming-pool': 0.19480794036008758, 'storage-tank': 0.6686508548425574, 'plane': 0.7888634108010129, 'large-vehicle': 0.3453482245636352, 'helicopter': 0.23330003330003332, 'harbor': 0.2005924600120569, 'basketball-court': 0.5268079153410501, 'bridge': 0.13876818973906352, 'small-vehicle': 0.3891952972340609, 'ground-track-field': 0.49036410617805964, 'soccer-ball-field': 0.5344762284410977, 'baseball-diamond': 0.4979394060721322, 'ship': 0.5183064069969582, 'tennis-court': 0.8989554000716976}}

"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20200919'
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


