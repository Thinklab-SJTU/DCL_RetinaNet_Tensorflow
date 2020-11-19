# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 256. + redundancy (2)
FLOPs: 876133269;    Trainable params: 33438561
{'0.75': {'ground-track-field': 0.324169238472122, 'small-vehicle': 0.22878003560927582, 'mAP': 0.3195218964764564, 'baseball-diamond': 0.2308962438956098, 'swimming-pool': 0.049272250059651634, 'ship': 0.30316383433715055, 'large-vehicle': 0.13499853709747145, 'helicopter': 0.10243584167029143, 'plane': 0.6291425883628943, 'harbor': 0.11170528817587641, 'basketball-court': 0.4244361704517376, 'tennis-court': 0.7946528444409552, 'storage-tank': 0.5719389755499619, 'roundabout': 0.3692319326463575, 'soccer-ball-field': 0.4184471442535958, 'bridge': 0.09955752212389381}, '0.8': {'ground-track-field': 0.17980929224948364, 'small-vehicle': 0.08103021916089725, 'mAP': 0.1949912314409527, 'baseball-diamond': 0.07575757575757576, 'swimming-pool': 0.0303030303030303, 'ship': 0.15735584081679005, 'large-vehicle': 0.0627284778853493, 'helicopter': 0.014545454545454545, 'plane': 0.39789301759992857, 'harbor': 0.015265436318067897, 'basketball-court': 0.2878787878787879, 'tennis-court': 0.6729225908420803, 'storage-tank': 0.4212338574798436, 'roundabout': 0.20331909268917142, 'soccer-ball-field': 0.2642197374817696, 'bridge': 0.0606060606060606}, '0.85': {'ground-track-field': 0.022727272727272728, 'small-vehicle': 0.045454545454545456, 'mAP': 0.10436306551670295, 'baseball-diamond': 0.033071018201129725, 'swimming-pool': 0.0013568521031207597, 'ship': 0.09974208068194267, 'large-vehicle': 0.006620046620046619, 'helicopter': 0.006993006993006993, 'plane': 0.21646295843729393, 'harbor': 0.00455684666210982, 'basketball-court': 0.11005832977663962, 'tennis-court': 0.5203268422317102, 'storage-tank': 0.2092860256505159, 'roundabout': 0.10457963089542038, 'soccer-ball-field': 0.13875598086124402, 'bridge': 0.045454545454545456}, 'mmAP': 0.34230222419137846, '0.7': {'ground-track-field': 0.4791258431100739, 'small-vehicle': 0.35659380196369306, 'mAP': 0.42389042243047903, 'baseball-diamond': 0.39533075571382026, 'swimming-pool': 0.11397878686585722, 'ship': 0.4893727814390966, 'large-vehicle': 0.2686086155841882, 'helicopter': 0.2429590017825312, 'plane': 0.7682536599425802, 'harbor': 0.15494616063548103, 'basketball-court': 0.47086078173034696, 'tennis-court': 0.8114562835262208, 'storage-tank': 0.6776726358904755, 'roundabout': 0.49254426346902835, 'soccer-ball-field': 0.5135760417268682, 'bridge': 0.12307692307692308}, '0.6': {'ground-track-field': 0.554491595457835, 'small-vehicle': 0.5345396255375449, 'mAP': 0.5744768107929559, 'baseball-diamond': 0.6177190467704075, 'swimming-pool': 0.39870022244543063, 'ship': 0.7133281826793233, 'large-vehicle': 0.46151652468399457, 'helicopter': 0.40378836645274435, 'plane': 0.8928194399349533, 'harbor': 0.3232211710814504, 'basketball-court': 0.5493827160493827, 'tennis-court': 0.9017780426512431, 'storage-tank': 0.7804636880927003, 'roundabout': 0.6266217457770006, 'soccer-ball-field': 0.6172749391371092, 'bridge': 0.2415068551432188}, '0.5': {'ground-track-field': 0.5769758669685272, 'small-vehicle': 0.5883863391235262, 'mAP': 0.6421948245310393, 'baseball-diamond': 0.7021594544414026, 'swimming-pool': 0.4963879248522083, 'ship': 0.7538601718681234, 'large-vehicle': 0.578066824261126, 'helicopter': 0.5147005350480178, 'plane': 0.8981076647609824, 'harbor': 0.5259628390816662, 'basketball-court': 0.5641548720915146, 'tennis-court': 0.9043272355959823, 'storage-tank': 0.7892841168458299, 'roundabout': 0.6727412744612636, 'soccer-ball-field': 0.687949013377617, 'bridge': 0.3798582351878026}, '0.65': {'ground-track-field': 0.5160866719046987, 'small-vehicle': 0.4580279137432583, 'mAP': 0.5066540436563701, 'baseball-diamond': 0.5096146337355681, 'swimming-pool': 0.2565384861196808, 'ship': 0.6137717786343888, 'large-vehicle': 0.3642333024614267, 'helicopter': 0.2962644820631796, 'plane': 0.848406004744934, 'harbor': 0.22920925315728097, 'basketball-court': 0.5054577398043614, 'tennis-court': 0.9002789912917637, 'storage-tank': 0.7627000337258805, 'roundabout': 0.5825588783179009, 'soccer-ball-field': 0.5917714173463517, 'bridge': 0.16489106779487905}, '0.9': {'ground-track-field': 0.007575757575757575, 'small-vehicle': 0.045454545454545456, 'mAP': 0.036819316833222024, 'baseball-diamond': 0.012987012987012986, 'swimming-pool': 0.00033921302578018993, 'ship': 0.09090909090909091, 'large-vehicle': 0.0012368583797155224, 'helicopter': 0.0, 'plane': 0.07847765488715043, 'harbor': 0.0017292490118577077, 'basketball-court': 0.014141414141414142, 'tennis-court': 0.17117124157400154, 'storage-tank': 0.06665155293584221, 'roundabout': 0.025252525252525252, 'soccer-ball-field': 0.03636363636363637, 'bridge': 0.0}, '0.95': {'ground-track-field': 0.0, 'small-vehicle': 0.00046860356138706655, 'mAP': 0.0024790490711042034, 'baseball-diamond': 0.0, 'swimming-pool': 0.0, 'ship': 0.0002029220779220779, 'large-vehicle': 0.0, 'helicopter': 0.0, 'plane': 0.01515151515151515, 'harbor': 0.0, 'basketball-court': 0.0, 'tennis-court': 0.006211180124223603, 'storage-tank': 0.01515151515151515, 'roundabout': 0.0, 'soccer-ball-field': 0.0, 'bridge': 0.0}, '0.55': {'ground-track-field': 0.5689169419635496, 'small-vehicle': 0.5725251337716407, 'mAP': 0.6176315811645022, 'baseball-diamond': 0.6949073804974161, 'swimming-pool': 0.45694672197809244, 'ship': 0.7425598166428333, 'large-vehicle': 0.5450439677469803, 'helicopter': 0.4663610898905017, 'plane': 0.8967407049204068, 'harbor': 0.42253452464065294, 'basketball-court': 0.5611472805834925, 'tennis-court': 0.9040468953952837, 'storage-tank': 0.7856249814937445, 'roundabout': 0.6640305571062819, 'soccer-ball-field': 0.6767548264512211, 'bridge': 0.3063328943854376}}

"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20201011'
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


