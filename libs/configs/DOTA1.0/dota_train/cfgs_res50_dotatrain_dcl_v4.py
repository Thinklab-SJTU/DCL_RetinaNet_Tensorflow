# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 128.

{'0.55': {'small-vehicle': 0.5713137682410476, 'helicopter': 0.427692604323442, 'tennis-court': 0.9046643133300374, 'baseball-diamond': 0.6753161308676068, 'bridge': 0.31671822183517206, 'ship': 0.756738113624651, 'roundabout': 0.66112249331332, 'ground-track-field': 0.5992945674219691, 'mAP': 0.6243366586833333, 'soccer-ball-field': 0.7049001870723182, 'plane': 0.8956196387528836, 'harbor': 0.47016040395187675, 'swimming-pool': 0.46275757693763736, 'basketball-court': 0.5658905058079504, 'storage-tank': 0.7864322277117192, 'large-vehicle': 0.5664291270583705},
'0.85': {'small-vehicle': 0.022727272727272728, 'helicopter': 0.018181818181818184, 'tennis-court': 0.5228734066141131, 'baseball-diamond': 0.12215909090909091, 'bridge': 0.09090909090909091, 'ship': 0.03272426410916533, 'roundabout': 0.0640074211502783, 'ground-track-field': 0.06598240469208211, 'mAP': 0.10907245321120428, 'soccer-ball-field': 0.12272727272727273, 'plane': 0.2355642415150327, 'harbor': 0.009090909090909092, 'swimming-pool': 0.0034965034965034965, 'basketball-court': 0.08363636363636363, 'storage-tank': 0.23290030699188244, 'large-vehicle': 0.00910643141718839},
'mmAP': 0.3568936617733111,
'0.9': {'small-vehicle': 0.0023986567522187576, 'helicopter': 0.0, 'tennis-court': 0.2327027301077516, 'baseball-diamond': 0.004784688995215311, 'bridge': 0.0, 'ship': 0.0030303030303030303, 'roundabout': 0.045454545454545456, 'ground-track-field': 0.006993006993006993, 'mAP': 0.02779386261686322, 'soccer-ball-field': 0.009090909090909092, 'plane': 0.04349512315128361, 'harbor': 0.0009039256198347108, 'swimming-pool': 0.0006447453255963894, 'basketball-court': 0.009828009828009828, 'storage-tank': 0.05344906349931476, 'large-vehicle': 0.004132231404958678},
'0.8': {'small-vehicle': 0.08516560231980108, 'helicopter': 0.028708133971291863, 'tennis-court': 0.7581570996592588, 'baseball-diamond': 0.22105244687757716, 'bridge': 0.09090909090909091, 'ship': 0.149317636090249, 'roundabout': 0.22304170739654613, 'ground-track-field': 0.22482951830777914, 'mAP': 0.22931366849052903, 'soccer-ball-field': 0.32285074857231055, 'plane': 0.5125963273758752, 'harbor': 0.0303030303030303, 'swimming-pool': 0.012326656394453005, 'basketball-court': 0.2863918793195109, 'storage-tank': 0.413654726206918, 'large-vehicle': 0.08040042365424338},
'0.65': {'small-vehicle': 0.46958891362388105, 'helicopter': 0.3591444440726737, 'tennis-court': 0.9025869525168013, 'baseball-diamond': 0.5694730789205221, 'bridge': 0.19561516805799858, 'ship': 0.6339285528702248, 'roundabout': 0.5637883624267427, 'ground-track-field': 0.5319965662537672, 'mAP': 0.5359448916803379, 'soccer-ball-field': 0.6377624204426712, 'plane': 0.8635707301468887, 'harbor': 0.28125757113505945, 'swimming-pool': 0.28476146324130786, 'basketball-court': 0.5396048113035694, 'storage-tank': 0.758558070196676, 'large-vehicle': 0.4475362699962861},
'0.75': {'small-vehicle': 0.20662164694895943, 'helicopter': 0.16655011655011656, 'tennis-court': 0.7982639564861062, 'baseball-diamond': 0.28428991502131323, 'bridge': 0.10805422647527911, 'ship': 0.3550072952532795, 'roundabout': 0.31609777786992976, 'ground-track-field': 0.39647876561366296, 'mAP': 0.342807692292932, 'soccer-ball-field': 0.486468188674071, 'plane': 0.6609227357026716, 'harbor': 0.11566985645933014, 'swimming-pool': 0.06486210418794688, 'basketball-court': 0.4573149237580861, 'storage-tank': 0.5534712786273144, 'large-vehicle': 0.17204259676591352},
'0.95': {'small-vehicle': 0.0, 'helicopter': 0.0, 'tennis-court': 0.09090909090909091, 'baseball-diamond': 0.0, 'bridge': 0.0, 'ship': 0.000122684333210649, 'roundabout': 0.0, 'ground-track-field': 0.0, 'mAP': 0.006408120321857617, 'soccer-ball-field': 0.0, 'plane': 0.0030303030303030303, 'harbor': 0.0, 'swimming-pool': 0.0, 'basketball-court': 0.0, 'storage-tank': 0.0019264287680029355, 'large-vehicle': 0.00013329778725673153},
'0.6': {'small-vehicle': 0.5291203646935283, 'helicopter': 0.4150701733092289, 'tennis-court': 0.9042632438113207, 'baseball-diamond': 0.6373920728605683, 'bridge': 0.2761297224357826, 'ship': 0.7368960625604253, 'roundabout': 0.6251554771887137, 'ground-track-field': 0.5771706321911241, 'mAP': 0.595512160744591, 'soccer-ball-field': 0.695757982760669, 'plane': 0.8919005382904578, 'harbor': 0.384311007055127, 'swimming-pool': 0.3991977591867879, 'basketball-court': 0.55641602083753, 'storage-tank': 0.7804706736643219, 'large-vehicle': 0.5234306803232815},
'0.7': {'small-vehicle': 0.34081619716388295, 'helicopter': 0.27293583609373084, 'tennis-court': 0.8905133974550816, 'baseball-diamond': 0.4149925806046143, 'bridge': 0.14213364241080428, 'ship': 0.5176504693039009, 'roundabout': 0.4243342516069789, 'ground-track-field': 0.4559030514620176, 'mAP': 0.44632016713704636, 'soccer-ball-field': 0.5963878475476413, 'plane': 0.779077669375429, 'harbor': 0.16812187812187812, 'swimming-pool': 0.1600759682349791, 'basketball-court': 0.5211085416500827, 'storage-tank': 0.6686520362410655, 'large-vehicle': 0.3420991397836083},
'0.5': {'small-vehicle': 0.5842472329043403, 'helicopter': 0.45610470171184453, 'tennis-court': 0.9046643133300374, 'baseball-diamond': 0.6946834656630617, 'bridge': 0.36763268910984387, 'ship': 0.8252109570249702, 'roundabout': 0.6725539292275144, 'ground-track-field': 0.6149872692998241, 'mAP': 0.6514269425544162, 'soccer-ball-field': 0.7121786995004279, 'plane': 0.8978604271577529, 'harbor': 0.5372539544680321, 'swimming-pool': 0.5005467913333125, 'basketball-court': 0.5658905058079504, 'storage-tank': 0.8240522031854918, 'large-vehicle': 0.6135369985918406}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20200917'
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


