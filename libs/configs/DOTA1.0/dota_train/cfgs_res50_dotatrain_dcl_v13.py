# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
GCL + OMEGA = 180 / 4.

{'0.75': {'large-vehicle': 0.05877629483008387, 'harbor': 0.02837327076979729, 'ship': 0.0770544317858896, 'helicopter': 0.08856361030274074, 'plane': 0.642756010227012, 'mAP': 0.23828958471773334, 'tennis-court': 0.5203937333944969, 'baseball-diamond': 0.23854426619132502, 'roundabout': 0.31291752544021145, 'swimming-pool': 0.06620074840413823, 'ground-track-field': 0.17624249541560952, 'soccer-ball-field': 0.32191142191142197, 'small-vehicle': 0.13586593270940245, 'basketball-court': 0.3417962179595191, 'storage-tank': 0.538209843509914, 'bridge': 0.026737967914438502},
'0.7': {'large-vehicle': 0.12419804962486619, 'harbor': 0.06978756535826326, 'ship': 0.1371977298966926, 'helicopter': 0.20450736289245605, 'plane': 0.7926651474951057, 'mAP': 0.34413340376551926, 'tennis-court': 0.7642432238286292, 'baseball-diamond': 0.35214075546870216, 'roundabout': 0.4251694382379541, 'swimming-pool': 0.11959846916368655, 'ground-track-field': 0.33837196028920286, 'soccer-ball-field': 0.44264738335884585, 'small-vehicle': 0.24671146971732033, 'basketball-court': 0.428694077109389, 'storage-tank': 0.6578884076964435, 'bridge': 0.058180016345231886},
'0.95': {'large-vehicle': 0.0001436162573603332, 'harbor': 0.0, 'ship': 0.0, 'helicopter': 0.0, 'plane': 0.045454545454545456, 'mAP': 0.005627683772235435, 'tennis-court': 0.009404388714733543, 'baseball-diamond': 0.0, 'roundabout': 0.007575757575757575, 'swimming-pool': 0.0, 'ground-track-field': 0.0, 'soccer-ball-field': 0.013986013986013986, 'small-vehicle': 5.872680291284943e-05, 'basketball-court': 0.0, 'storage-tank': 0.007792207792207792, 'bridge': 0.0},
'0.6': {'large-vehicle': 0.29837642625959176, 'harbor': 0.22207116566896862, 'ship': 0.4018016173342976, 'helicopter': 0.40057001535839637, 'plane': 0.8952628884662741, 'mAP': 0.5402694055450408, 'tennis-court': 0.9070180635975226, 'baseball-diamond': 0.6381520889312664, 'roundabout': 0.5908037785529748, 'swimming-pool': 0.44504330410398696, 'ground-track-field': 0.5788980949405267, 'soccer-ball-field': 0.6487365125296158, 'small-vehicle': 0.5174590512918251, 'basketball-court': 0.5930172137917984, 'storage-tank': 0.7771041756971718, 'bridge': 0.18972668665139503},
'0.65': {'large-vehicle': 0.20714698630943099, 'harbor': 0.11775562072336265, 'ship': 0.22537193763352636, 'helicopter': 0.2721730646384796, 'plane': 0.8876231565229372, 'mAP': 0.4593513563373495, 'tennis-court': 0.9050973839576681, 'baseball-diamond': 0.5713644874621899, 'roundabout': 0.5487908626981037, 'swimming-pool': 0.28154259574685025, 'ground-track-field': 0.5113601532073486, 'soccer-ball-field': 0.5431753186042907, 'small-vehicle': 0.4132625466287009, 'basketball-court': 0.5451116127044414, 'storage-tank': 0.7587595871775359, 'bridge': 0.10173503104537587},
'0.5': {'large-vehicle': 0.4830164220317368, 'harbor': 0.41196618509663846, 'ship': 0.7293408220138738, 'helicopter': 0.47304105881262537, 'plane': 0.8984174489752382, 'mAP': 0.6297712005896201, 'tennis-court': 0.9084224598930483, 'baseball-diamond': 0.7083128007406175, 'roundabout': 0.6605913092662579, 'swimming-pool': 0.5320806116970682, 'ground-track-field': 0.6159503832165075, 'soccer-ball-field': 0.6894517592009755, 'small-vehicle': 0.601645585263301, 'basketball-court': 0.6389655169943154, 'storage-tank': 0.7863340097046186, 'bridge': 0.3090316359374794},
'0.8': {'large-vehicle': 0.019844880599628904, 'harbor': 0.008116883116883118, 'ship': 0.03128501137474028, 'helicopter': 0.03896103896103896, 'plane': 0.4278794893717084, 'mAP': 0.1547783394218723, 'tennis-court': 0.48669780018080133, 'baseball-diamond': 0.11898395721925134, 'roundabout': 0.18231763224746383, 'swimming-pool': 0.045454545454545456, 'ground-track-field': 0.10108485409132958, 'soccer-ball-field': 0.24965034965034966, 'small-vehicle': 0.03641125619046692, 'basketball-court': 0.22215041632566113, 'storage-tank': 0.34071576442300394, 'bridge': 0.012121212121212121},
'mmAP': 0.3080817206767284,
'0.9': {'large-vehicle': 0.000825003966365223, 'harbor': 0.0002331002331002331, 'ship': 0.0023923444976076554, 'helicopter': 0.0, 'plane': 0.09363914585796758, 'mAP': 0.03621870218765814, 'tennis-court': 0.12817489735011475, 'baseball-diamond': 0.045454545454545456, 'roundabout': 0.03636363636363637, 'swimming-pool': 0.0, 'ground-track-field': 0.002932551319648094, 'soccer-ball-field': 0.09090909090909091, 'small-vehicle': 0.001569976076555024, 'basketball-court': 0.09090909090909091, 'storage-tank': 0.04987714987714988, 'bridge': 0.0},
'0.85': {'large-vehicle': 0.003998096144693004, 'harbor': 0.002525252525252525, 'ship': 0.007272727272727273, 'helicopter': 0.01515151515151515, 'plane': 0.2427880645523664, 'mAP': 0.08146319281071593, 'tennis-court': 0.3239103995675737, 'baseball-diamond': 0.0606060606060606, 'roundabout': 0.06842105263157895, 'swimming-pool': 0.0007736943907156673, 'ground-track-field': 0.009090909090909092, 'soccer-ball-field': 0.17922526911290956, 'small-vehicle': 0.01515151515151515, 'basketball-court': 0.11261872455902307, 'storage-tank': 0.1764620422339381, 'bridge': 0.003952569169960474},
'0.55': {'large-vehicle': 0.3923435132107821, 'harbor': 0.346191249211942, 'ship': 0.5634813084085719, 'helicopter': 0.423653726953639, 'plane': 0.8978159176030857, 'mAP': 0.5909143376195393, 'tennis-court': 0.9074707122113594, 'baseball-diamond': 0.6917546604151645, 'roundabout': 0.6337263635351447, 'swimming-pool': 0.507687290094447, 'ground-track-field': 0.6032323465475056, 'soccer-ball-field': 0.6662066686361356, 'small-vehicle': 0.5817774679359003, 'basketball-court': 0.6285001594805458, 'storage-tank': 0.7837597641028997, 'bridge': 0.2361139159459653}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_G_2x_20200927'
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


