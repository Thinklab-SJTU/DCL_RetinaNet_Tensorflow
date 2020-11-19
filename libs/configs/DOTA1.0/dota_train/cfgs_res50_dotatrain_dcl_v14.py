# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
GCL + OMEGA = 180 / 8.

{'0.75': {'ground-track-field': 0.37705769205661493, 'bridge': 0.045454545454545456, 'plane': 0.7503264080250396, 'roundabout': 0.29367110497789356, 'baseball-diamond': 0.2454109165868756, 'storage-tank': 0.5539121498409814, 'basketball-court': 0.46654407203932313, 'mAP': 0.33919064043451025, 'tennis-court': 0.8716821805280958, 'ship': 0.363671088513871, 'soccer-ball-field': 0.47122704484455635, 'harbor': 0.050834073635376564, 'swimming-pool': 0.11152764761012184, 'large-vehicle': 0.11618281598480452, 'small-vehicle': 0.23294901858289782, 'helicopter': 0.13740884783665533},
'0.6': {'ground-track-field': 0.5306653907618285, 'bridge': 0.24173689989584768, 'plane': 0.889494636570939, 'roundabout': 0.5617879652218779, 'baseball-diamond': 0.63498712707518, 'storage-tank': 0.7837189837099175, 'basketball-court': 0.5462501257893673, 'mAP': 0.5846733872234863, 'tennis-court': 0.9082690476190478, 'ship': 0.7407349660461138, 'soccer-ball-field': 0.6796878419963602, 'harbor': 0.31963711299577163, 'swimming-pool': 0.42204463254818825, 'large-vehicle': 0.47771041962363947, 'small-vehicle': 0.5594485497209403, 'helicopter': 0.47392710877727523},
'0.8': {'ground-track-field': 0.18533467258957453, 'bridge': 0.045454545454545456, 'plane': 0.5043134399249637, 'roundabout': 0.20276543631806793, 'baseball-diamond': 0.12343575992028602, 'storage-tank': 0.38292829908090215, 'basketball-court': 0.3822933637449767, 'mAP': 0.2296141955593481, 'tennis-court': 0.7794174132036972, 'ship': 0.13720528894003337, 'soccer-ball-field': 0.39122441545762543, 'harbor': 0.011461318051575931, 'swimming-pool': 0.09090909090909091, 'large-vehicle': 0.030904923953052298, 'small-vehicle': 0.08760392688079124, 'helicopter': 0.08896103896103896},
'0.85': {'ground-track-field': 0.009404388714733543, 'bridge': 0.0006887052341597796, 'plane': 0.25115319555418053, 'roundabout': 0.06578947368421052, 'baseball-diamond': 0.0606060606060606, 'storage-tank': 0.14284820667727016, 'basketball-court': 0.10305823209049017, 'mAP': 0.10423805496954416, 'tennis-court': 0.5388914169409865, 'ship': 0.030701304224357705, 'soccer-ball-field': 0.18867163878250354, 'harbor': 0.003125814014066163, 'swimming-pool': 0.09090909090909091, 'large-vehicle': 0.006294725682480785, 'small-vehicle': 0.025974025974025972, 'helicopter': 0.045454545454545456},
'0.5': {'ground-track-field': 0.5735155608849527, 'bridge': 0.3687554858910357, 'plane': 0.8940940697231514, 'roundabout': 0.6585612497886228, 'baseball-diamond': 0.7288212755012599, 'storage-tank': 0.7901351413815751, 'basketball-court': 0.5648987430512358, 'mAP': 0.6523361638412574, 'tennis-court': 0.9084103524287575, 'ship': 0.8240221811515541, 'soccer-ball-field': 0.7001629593291009, 'harbor': 0.5110234037284048, 'swimming-pool': 0.5325293900368956, 'large-vehicle': 0.5981379356729946, 'small-vehicle': 0.6156555743136127, 'helicopter': 0.516319134735707},
'0.7': {'ground-track-field': 0.4616976024925334, 'bridge': 0.10014781966001479, 'plane': 0.790341745111216, 'roundabout': 0.42324079524844527, 'baseball-diamond': 0.4502474968920562, 'storage-tank': 0.6761750378569069, 'basketball-court': 0.5300699276847941, 'mAP': 0.43563945225135814, 'tennis-court': 0.9070285381418742, 'ship': 0.5127349012709608, 'soccer-ball-field': 0.5643223965165349, 'harbor': 0.12751017164653528, 'swimming-pool': 0.19408310511717203, 'large-vehicle': 0.22077623505340135, 'small-vehicle': 0.37651264078632085, 'helicopter': 0.1997033702916056},
'0.95': {'ground-track-field': 0.0, 'bridge': 0.0, 'plane': 0.0025974025974025974, 'roundabout': 0.0, 'baseball-diamond': 0.0, 'storage-tank': 0.006060606060606061, 'basketball-court': 0.0, 'mAP': 0.0027580850522026996, 'tennis-court': 0.01948051948051948, 'ship': 0.0017825311942959, 'soccer-ball-field': 0.011363636363636364, 'harbor': 0.0, 'swimming-pool': 0.0, 'large-vehicle': 8.658008658008658e-05, 'small-vehicle': 0.0, 'helicopter': 0.0},
'0.55': {'ground-track-field': 0.5458842345418469, 'bridge': 0.3106861203207727, 'plane': 0.8930420931179962, 'roundabout': 0.6405772229922218, 'baseball-diamond': 0.6727377484379607, 'storage-tank': 0.7877114916185537, 'basketball-court': 0.5539065545493779, 'mAP': 0.6208718131925184, 'tennis-court': 0.9082690476190478, 'ship': 0.7562905510710827, 'soccer-ball-field': 0.6848292085094736, 'harbor': 0.42080371880309436, 'swimming-pool': 0.49030316135251845, 'large-vehicle': 0.5428089440193388, 'small-vehicle': 0.6009274850975838, 'helicopter': 0.5042996158369074},
'0.9': {'ground-track-field': 0.004329004329004329, 'bridge': 0.0, 'plane': 0.05783405783405784, 'roundabout': 0.045454545454545456, 'baseball-diamond': 0.0606060606060606, 'storage-tank': 0.03654183348849429, 'basketball-court': 0.0303030303030303, 'mAP': 0.03749893357363957, 'tennis-court': 0.1905185072600853, 'ship': 0.00505050505050505, 'soccer-ball-field': 0.125, 'harbor': 0.0008391608391608393, 'swimming-pool': 0.000368052999631947, 'large-vehicle': 0.0013911570797798343, 'small-vehicle': 0.004248088360237893, 'helicopter': 0.0},
'0.65': {'ground-track-field': 0.4977702027269791, 'bridge': 0.1508770776530151, 'plane': 0.8815972499777556, 'roundabout': 0.5447985092960335, 'baseball-diamond': 0.5704423683475066, 'storage-tank': 0.7654915095016543, 'basketball-court': 0.5462501257893673, 'mAP': 0.5224202851085369, 'tennis-court': 0.9082680395610352, 'ship': 0.6372098174070326, 'soccer-ball-field': 0.647583078700947, 'harbor': 0.2297375068146538, 'swimming-pool': 0.29283143966377667, 'large-vehicle': 0.3791291985229039, 'small-vehicle': 0.5082314498387233, 'helicopter': 0.27608670282667047},
'mmAP': 0.35292410112064015}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_G_2x_20200928'
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


