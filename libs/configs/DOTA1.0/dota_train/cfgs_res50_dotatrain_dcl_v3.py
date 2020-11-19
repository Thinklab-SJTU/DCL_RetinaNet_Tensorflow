# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 512.

{'0.6': {'helicopter': 0.38701955173036706, 'harbor': 0.3841116316127199, 'ground-track-field': 0.578710934391982, 'bridge': 0.26417703557895683, 'roundabout': 0.614854283394806, 'basketball-court': 0.5122921458678094, 'storage-tank': 0.7815425591785349, 'small-vehicle': 0.5549670465472254, 'tennis-court': 0.9057066042099879, 'swimming-pool': 0.4086701308880284, 'baseball-diamond': 0.630190791066799, 'ship': 0.7192497165577004, 'soccer-ball-field': 0.6606301896907484, 'mAP': 0.5843047513834546, 'plane': 0.891308602470951, 'large-vehicle': 0.4711400475652019},
'0.5': {'helicopter': 0.442460192312206, 'harbor': 0.5385492899104309, 'ground-track-field': 0.636053022133887, 'bridge': 0.37925028577825276, 'roundabout': 0.6743134021804009, 'basketball-court': 0.5278512180755465, 'storage-tank': 0.8150156088215127, 'small-vehicle': 0.6087326276641921, 'tennis-court': 0.9072596004154013, 'swimming-pool': 0.5143731872078271, 'baseball-diamond': 0.699480748504615, 'ship': 0.8003638915948252, 'soccer-ball-field': 0.7052707957410069, 'mAP': 0.6487955415076578, 'plane': 0.8977616870062499, 'large-vehicle': 0.5851975652685126},
'0.85': {'helicopter': 0.03409090909090909, 'harbor': 0.012121212121212121, 'ground-track-field': 0.052297165200391, 'bridge': 0.018181818181818184, 'roundabout': 0.11468531468531469, 'basketball-court': 0.11621174524400331, 'storage-tank': 0.1764406601097195, 'small-vehicle': 0.0303030303030303, 'tennis-court': 0.5156835889871876, 'swimming-pool': 0.003116883116883117, 'baseball-diamond': 0.05472027972027972, 'ship': 0.037058536091264166, 'soccer-ball-field': 0.15169082125603864, 'mAP': 0.10106529133835274, 'plane': 0.19173339832323197, 'large-vehicle': 0.0076440076440076445},
'0.7': {'helicopter': 0.2346899090214598, 'harbor': 0.1706790099324717, 'ground-track-field': 0.4731174757240437, 'bridge': 0.1343339203378808, 'roundabout': 0.4772259013625298, 'basketball-court': 0.4568745061609755, 'storage-tank': 0.6760246020643736, 'small-vehicle': 0.3916907340979572, 'tennis-court': 0.900527062039714, 'swimming-pool': 0.18541940408826305, 'baseball-diamond': 0.46539213231261783, 'ship': 0.4859798765025512, 'soccer-ball-field': 0.5699167511326699, 'mAP': 0.44497959628157857, 'plane': 0.7812370266526443, 'large-vehicle': 0.2715856327935252},
'0.55': {'helicopter': 0.42402975266352877, 'harbor': 0.48073747272198564, 'ground-track-field': 0.6124543213210784, 'bridge': 0.3242590969509706, 'roundabout': 0.6571360281068076, 'basketball-court': 0.5235589560364673, 'storage-tank': 0.7894168226937517, 'small-vehicle': 0.5935321828511616, 'tennis-court': 0.9067183430745986, 'swimming-pool': 0.4820163868773806, 'baseball-diamond': 0.6757210070516295, 'ship': 0.7424059680529825, 'soccer-ball-field': 0.6822883128143064, 'mAP': 0.6210226416801916, 'plane': 0.8961386497093871, 'large-vehicle': 0.5249263242768368},
'mmAP': 0.34993488738128026,
'0.75': {'helicopter': 0.16333410834689607, 'harbor': 0.05510452221848308, 'ground-track-field': 0.33538305129214224, 'bridge': 0.09090909090909091, 'roundabout': 0.32034146409146413, 'basketball-court': 0.4067197100285335, 'storage-tank': 0.5603710777627343, 'small-vehicle': 0.24228708684955252, 'tennis-court': 0.8074190545383705, 'swimming-pool': 0.07667882213919386, 'baseball-diamond': 0.2894926164423383, 'ship': 0.31843864901241203, 'soccer-ball-field': 0.5093138189039557, 'mAP': 0.33088127784197063, 'plane': 0.6485023375443284, 'large-vehicle': 0.1389237575500641},
'0.9': {'helicopter': 0.01515151515151515, 'harbor': 0.0011730205278592375, 'ground-track-field': 0.009569377990430622, 'bridge': 0.0032467532467532465, 'roundabout': 0.0303030303030303, 'basketball-court': 0.013636363636363636, 'storage-tank': 0.022727272727272728, 'small-vehicle': 0.012121212121212121, 'tennis-court': 0.20701300874767398, 'swimming-pool': 0.0006184291898577612, 'baseball-diamond': 0.025974025974025972, 'ship': 0.004217432052483599, 'soccer-ball-field': 0.020527859237536656, 'mAP': 0.027586564327429355, 'plane': 0.046044962531223976, 'large-vehicle': 0.0014742014742014744},
'0.95': {'helicopter': 0.0, 'harbor': 0.0, 'ground-track-field': 0.0, 'bridge': 0.0, 'roundabout': 0.0036363636363636364, 'basketball-court': 0.0016233766233766233, 'storage-tank': 0.011363636363636364, 'small-vehicle': 0.0009276437847866418, 'tennis-court': 0.014799154334038056, 'swimming-pool': 0.0, 'baseball-diamond': 0.0, 'ship': 5.8688890193086445e-05, 'soccer-ball-field': 0.0, 'mAP': 0.002412574466803543, 'plane': 0.0036363636363636364, 'large-vehicle': 0.00014338973329509606},
'0.65': {'helicopter': 0.3325111317046801, 'harbor': 0.27424382175093753, 'ground-track-field': 0.5558291546637624, 'bridge': 0.20462341452568664, 'roundabout': 0.5711564153986403, 'basketball-court': 0.48953723371715824, 'storage-tank': 0.7602335033782641, 'small-vehicle': 0.49757874355798115, 'tennis-court': 0.9057066042099879, 'swimming-pool': 0.27615984588225057, 'baseball-diamond': 0.5174977181780882, 'ship': 0.6162937428784862, 'soccer-ball-field': 0.6434254564313463, 'mAP': 0.5275346501079196, 'plane': 0.8761158381509035, 'large-vehicle': 0.3921071271906202},
'0.8': {'helicopter': 0.03409090909090909, 'harbor': 0.018181818181818184, 'ground-track-field': 0.1820708477976117, 'bridge': 0.024242424242424242, 'roundabout': 0.17922400180464695, 'basketball-court': 0.2743817107262485, 'storage-tank': 0.38358096460955876, 'small-vehicle': 0.12194377232400046, 'tennis-court': 0.7693062944343074, 'swimming-pool': 0.007820136852394917, 'baseball-diamond': 0.16949458809923926, 'ship': 0.13538930476509353, 'soccer-ball-field': 0.4017745650575839, 'mAP': 0.2107659848774434, 'plane': 0.4204185147184285, 'large-vehicle': 0.03956992045738553}}

"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20200916'
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
OMEGA = 180 / 512.
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


