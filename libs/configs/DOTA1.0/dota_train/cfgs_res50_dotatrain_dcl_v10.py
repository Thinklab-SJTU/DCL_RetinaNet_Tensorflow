# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
GCL + OMEGA = 180 / 64.

{'0.85': {'large-vehicle': 0.026581641504624057, 'roundabout': 0.06382978723404255, 'bridge': 0.022727272727272728, 'ground-track-field': 0.12354312354312355, 'tennis-court': 0.6791003588868861, 'harbor': 0.045454545454545456, 'small-vehicle': 0.014167650531286895, 'swimming-pool': 0.006993006993006993, 'storage-tank': 0.14351417124531882, 'baseball-diamond': 0.09159519725557462, 'mAP': 0.12651095262970513, 'soccer-ball-field': 0.14500683994528044, 'plane': 0.2961135541763281, 'basketball-court': 0.17554858934169276, 'helicopter': 0.018181818181818184, 'ship': 0.04530673242477591},
'0.55': {'large-vehicle': 0.5539125863537635, 'roundabout': 0.6330112210173324, 'bridge': 0.3403906424817351, 'ground-track-field': 0.5888590576447404, 'tennis-court': 0.9070754504900537, 'harbor': 0.4808019375970591, 'small-vehicle': 0.5912979406620922, 'swimming-pool': 0.47764064832020964, 'storage-tank': 0.7867326361342927, 'baseball-diamond': 0.6739731020850721, 'mAP': 0.6205593542787738, 'soccer-ball-field': 0.6922742847134024, 'plane': 0.8962431150937196, 'basketball-court': 0.5626407145832281, 'helicopter': 0.3724309123181001, 'ship': 0.751106064686808},
'0.65': {'large-vehicle': 0.42240596921423446, 'roundabout': 0.5413120627894838, 'bridge': 0.196401228769122, 'ground-track-field': 0.5241750429433714, 'tennis-court': 0.9040606043929514, 'harbor': 0.29188453480986964, 'small-vehicle': 0.5049662925725917, 'swimming-pool': 0.3124074727930575, 'storage-tank': 0.7606527034311815, 'baseball-diamond': 0.5515478164731895, 'mAP': 0.5310655994729611, 'soccer-ball-field': 0.5767418124482888, 'plane': 0.8768709064918805, 'basketball-court': 0.529711356395398, 'helicopter': 0.2889909512012111, 'ship': 0.6838552373685842},
'0.7': {'large-vehicle': 0.3278225426023929, 'roundabout': 0.4329317110681748, 'bridge': 0.10189093856440315, 'ground-track-field': 0.47673032382596137, 'tennis-court': 0.9019456471243976, 'harbor': 0.20577614535103098, 'small-vehicle': 0.40083765721368936, 'swimming-pool': 0.21705171856632366, 'storage-tank': 0.6657870210279688, 'baseball-diamond': 0.32497109762333537, 'mAP': 0.4386075807275056, 'soccer-ball-field': 0.5106863472801333, 'plane': 0.7870480313501631, 'basketball-court': 0.5150447544208056, 'helicopter': 0.13250713051861968, 'ship': 0.5780826443751853},
'0.95': {'large-vehicle': 9.733307377846993e-05, 'roundabout': 0.0018939393939393938, 'bridge': 0.0, 'ground-track-field': 0.0, 'tennis-court': 0.0303030303030303, 'harbor': 0.0, 'small-vehicle': 6.230917814194032e-05, 'swimming-pool': 0.0, 'storage-tank': 0.0015879317189360857, 'baseball-diamond': 0.0, 'mAP': 0.004052677988920356, 'soccer-ball-field': 0.012987012987012986, 'plane': 0.0036496350364963502, 'basketball-court': 0.0101010101010101, 'helicopter': 0.0, 'ship': 0.00010796804145972792},
'0.75': {'large-vehicle': 0.22100424784192907, 'roundabout': 0.33895579669000897, 'bridge': 0.06366697275788184, 'ground-track-field': 0.38090050046571783, 'tennis-court': 0.8102762947872778, 'harbor': 0.12073188301397493, 'small-vehicle': 0.22676817393906248, 'swimming-pool': 0.0741344574087508, 'storage-tank': 0.5479788249555296, 'baseball-diamond': 0.19354786966989734, 'mAP': 0.3323276309321542, 'soccer-ball-field': 0.39762880435334774, 'plane': 0.6662613472337602, 'basketball-court': 0.4607082804574968, 'helicopter': 0.10250599840042655, 'ship': 0.3798450120072519},
'0.6': {'large-vehicle': 0.5170419021691184, 'roundabout': 0.6182555958190463, 'bridge': 0.26364642006150957, 'ground-track-field': 0.5663750380108511, 'tennis-court': 0.906168073108188, 'harbor': 0.3891657191574295, 'small-vehicle': 0.5517986276663429, 'swimming-pool': 0.4101783891750117, 'storage-tank': 0.7824341331508416, 'baseball-diamond': 0.6315774273059348, 'mAP': 0.5841806380450598, 'soccer-ball-field': 0.6355177944630045, 'plane': 0.8929562132265589, 'basketball-court': 0.5475243297618491, 'helicopter': 0.3150298302912475, 'ship': 0.7350400773089619},
'0.9': {'large-vehicle': 0.0033379147849990185, 'roundabout': 0.0202020202020202, 'bridge': 0.011363636363636364, 'ground-track-field': 0.0202020202020202, 'tennis-court': 0.3450612947488031, 'harbor': 0.0011556240369799693, 'small-vehicle': 0.0034965034965034965, 'swimming-pool': 0.006993006993006993, 'storage-tank': 0.02118933697881066, 'baseball-diamond': 0.0606060606060606, 'mAP': 0.04547029020740596, 'soccer-ball-field': 0.07897507897507897, 'plane': 0.07339258797234784, 'basketball-court': 0.0303030303030303, 'helicopter': 0.0013774104683195593, 'ship': 0.00439882697947214},
'mmAP': 0.35669142559607275,
'0.5': {'large-vehicle': 0.6058527195392924, 'roundabout': 0.6735338051913051, 'bridge': 0.3804732883849522, 'ground-track-field': 0.6133124961794663, 'tennis-court': 0.9084145021645024, 'harbor': 0.5436331708382021, 'small-vehicle': 0.6039175934190584, 'swimming-pool': 0.5165997699785341, 'storage-tank': 0.7896551460292471, 'baseball-diamond': 0.6986381703310772, 'mAP': 0.6478039341382175, 'soccer-ball-field': 0.7053613962182406, 'plane': 0.8967746481743797, 'basketball-court': 0.577914638994647, 'helicopter': 0.38457271185525455, 'ship': 0.8184049547751047},
'0.8': {'large-vehicle': 0.09074926358578164, 'roundabout': 0.22982740316073652, 'bridge': 0.0404040404040404, 'ground-track-field': 0.2481772613351561, 'tennis-court': 0.8038211963269032, 'harbor': 0.09090909090909091, 'small-vehicle': 0.08828681212593291, 'swimming-pool': 0.020416605719300814, 'storage-tank': 0.346187049523413, 'baseball-diamond': 0.14115884115884117, 'mAP': 0.23633559754002356, 'soccer-ball-field': 0.2824561403508772, 'plane': 0.5286733980371231, 'basketball-court': 0.3832972582972583, 'helicopter': 0.07021118829655563, 'ship': 0.1804584138693419}}

"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_G_2x_20200924'
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
OMEGA = 180 / 64.
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


