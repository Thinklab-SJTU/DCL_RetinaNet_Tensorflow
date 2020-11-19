# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
GCL + OMEGA = 180 / 128.

{'0.9': {'bridge': 0.00024770869457517957, 'small-vehicle': 0.01515151515151515, 'basketball-court': 0.022727272727272728, 'harbor': 0.002457002457002457, 'ground-track-field': 0.045454545454545456, 'swimming-pool': 0.00031133250311332503, 'tennis-court': 0.32212580430571647, 'ship': 0.0061250805931656995, 'helicopter': 0.0, 'large-vehicle': 0.018181818181818184, 'storage-tank': 0.0404040404040404, 'baseball-diamond': 0.03636363636363637, 'soccer-ball-field': 0.09640359640359641, 'roundabout': 0.03409090909090909, 'plane': 0.08991686779297398, 'mAP': 0.048664075341592054},
'mmAP': 0.36338605912769173,
'0.65': {'bridge': 0.20392479684296969, 'small-vehicle': 0.5057416359130765, 'basketball-court': 0.5980623274406961, 'harbor': 0.2978391304228673, 'ground-track-field': 0.4796782877617, 'swimming-pool': 0.3442742751481132, 'tennis-court': 0.9037904742896131, 'ship': 0.671645069701009, 'helicopter': 0.338410437804781, 'large-vehicle': 0.4216644577885507, 'storage-tank': 0.7604602517350043, 'baseball-diamond': 0.5388564774822115, 'soccer-ball-field': 0.538680179560163, 'roundabout': 0.5992998665519418, 'plane': 0.8860328133644406, 'mAP': 0.5392240321204759}, '0.5': {'bridge': 0.3773526503129489, 'small-vehicle': 0.6050949114207811, 'basketball-court': 0.6185221194749325, 'harbor': 0.5436155133719833, 'ground-track-field': 0.568461995820489, 'swimming-pool': 0.5263694523372056, 'tennis-court': 0.9082494850052, 'ship': 0.811488947799433, 'helicopter': 0.5622251754713393, 'large-vehicle': 0.5942822039380387, 'storage-tank': 0.8121374171147271, 'baseball-diamond': 0.7102485009655304, 'soccer-ball-field': 0.6782356091879903, 'roundabout': 0.7049303619852034, 'plane': 0.8987138190165549, 'mAP': 0.6613285442148238},
'0.75': {'bridge': 0.054656002080894786, 'small-vehicle': 0.22481097169330722, 'basketball-court': 0.42785750177054527, 'harbor': 0.13668888148671088, 'ground-track-field': 0.37115666932446106, 'swimming-pool': 0.05608521924155498, 'tennis-court': 0.8098271719578611, 'ship': 0.3980074619047884, 'helicopter': 0.13283751310263192, 'large-vehicle': 0.2159336510834698, 'storage-tank': 0.5468629576990298, 'baseball-diamond': 0.23561124561746763, 'soccer-ball-field': 0.46407206268908396, 'roundabout': 0.3098958645699641, 'plane': 0.6637815977697311, 'mAP': 0.33653898479943345},
'0.95': {'bridge': 0.0, 'small-vehicle': 0.0, 'basketball-court': 0.0037878787878787876, 'harbor': 0.0, 'ground-track-field': 0.0, 'swimming-pool': 0.0, 'tennis-court': 0.0606060606060606, 'ship': 0.0012626262626262625, 'helicopter': 0.0, 'large-vehicle': 0.0002070822116380203, 'storage-tank': 0.0101010101010101, 'baseball-diamond': 0.018181818181818184, 'soccer-ball-field': 0.0, 'roundabout': 0.022727272727272728, 'plane': 0.0101010101010101, 'mAP': 0.00846498393195432},
'0.8': {'bridge': 0.017316017316017316, 'small-vehicle': 0.08642935991943901, 'basketball-court': 0.2955643390425999, 'harbor': 0.10182450959713141, 'ground-track-field': 0.22443759943759942, 'swimming-pool': 0.009324009324009324, 'tennis-court': 0.790969572988414, 'ship': 0.18410266139927758, 'helicopter': 0.039311004784689, 'large-vehicle': 0.08682691636232377, 'storage-tank': 0.3921699477068735, 'baseball-diamond': 0.16707201556539772, 'soccer-ball-field': 0.3566948555320648, 'roundabout': 0.20085449575471748, 'plane': 0.5245212076478774, 'mAP': 0.23182790082522878},
'0.6': {'bridge': 0.2650417047342151, 'small-vehicle': 0.5498954983379757, 'basketball-court': 0.6025670637125745, 'harbor': 0.3880452219660909, 'ground-track-field': 0.5515762621138928, 'swimming-pool': 0.42931692211177486, 'tennis-court': 0.9067884215877937, 'ship': 0.7376600089162743, 'helicopter': 0.4478358975504907, 'large-vehicle': 0.5036668258854706, 'storage-tank': 0.780508910923584, 'baseball-diamond': 0.6309791455953906, 'soccer-ball-field': 0.6001169562145171, 'roundabout': 0.6294236268872663, 'plane': 0.8958429800136145, 'mAP': 0.5946176964367283},
'0.85': {'bridge': 0.017045454545454544, 'small-vehicle': 0.036290702051571616, 'basketball-court': 0.16846580328423905, 'harbor': 0.010263929618768328, 'ground-track-field': 0.11940627202255108, 'swimming-pool': 0.006060606060606061, 'tennis-court': 0.6305098038830778, 'ship': 0.0496763859595718, 'helicopter': 0.025974025974025972, 'large-vehicle': 0.030299290342486893, 'storage-tank': 0.18316034877901005, 'baseball-diamond': 0.08712121212121211, 'soccer-ball-field': 0.14398644833427443, 'roundabout': 0.08157330884603611, 'plane': 0.2773251258381906, 'mAP': 0.12447724784407176},
'0.7': {'bridge': 0.15501642188758755, 'small-vehicle': 0.3971578524152296, 'basketball-court': 0.5411329248692315, 'harbor': 0.20344951830896021, 'ground-track-field': 0.44305980839345743, 'swimming-pool': 0.22189605623611444, 'tennis-court': 0.900463795197538, 'ship': 0.5926790079993837, 'helicopter': 0.222977022977023, 'large-vehicle': 0.32265737324035937, 'storage-tank': 0.6647945101238526, 'baseball-diamond': 0.3823842713137583, 'soccer-ball-field': 0.5130517535303948, 'roundabout': 0.4876997389348267, 'plane': 0.7891463835136144, 'mAP': 0.4558377625960887},
'0.55': {'bridge': 0.3272316375507865, 'small-vehicle': 0.5905756730681081, 'basketball-court': 0.6185221194749325, 'harbor': 0.4918297623675775, 'ground-track-field': 0.5561835076332096, 'swimming-pool': 0.4921664512080145, 'tennis-court': 0.9082494850052, 'ship': 0.7514392358220188, 'helicopter': 0.5337358607747623, 'large-vehicle': 0.5408703724891462, 'storage-tank': 0.7836379714818542, 'baseball-diamond': 0.6954106978016167, 'soccer-ball-field': 0.6485177746203328, 'roundabout': 0.6573821974363171, 'plane': 0.8974377007639363, 'mAP': 0.6328793631665207}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_G_2x_20200926'
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
OMEGA = 180 / 128.
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


