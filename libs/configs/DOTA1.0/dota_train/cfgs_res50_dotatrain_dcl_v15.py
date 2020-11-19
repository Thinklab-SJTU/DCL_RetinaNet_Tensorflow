# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
GCL + OMEGA = 180 / 512.

{'0.6': {'ground-track-field': 0.573582489319409, 'harbor': 0.3891521609424017, 'bridge': 0.2563337419887201, 'small-vehicle': 0.5648505388890961, 'plane': 0.8953705097216129, 'baseball-diamond': 0.6304525425142407, 'tennis-court': 0.9068133847959017, 'roundabout': 0.5504477682851595, 'storage-tank': 0.7818913345802345, 'swimming-pool': 0.39985514157699587, 'mAP': 0.5792389738191542, 'soccer-ball-field': 0.624200360919821, 'basketball-court': 0.5216235844619704, 'large-vehicle': 0.5246429570098051, 'ship': 0.7314627227976299, 'helicopter': 0.3379053694843169},
'0.8': {'ground-track-field': 0.2640926811979444, 'harbor': 0.0994356798615974, 'bridge': 0.09090909090909091, 'small-vehicle': 0.14845898197949595, 'plane': 0.5189377689746963, 'baseball-diamond': 0.14224201616818288, 'tennis-court': 0.7850084962037644, 'roundabout': 0.2161224596513639, 'storage-tank': 0.4032224420253035, 'swimming-pool': 0.021645021645021644, 'mAP': 0.25175554640925113, 'soccer-ball-field': 0.38894355893358884, 'basketball-court': 0.361673373734271, 'large-vehicle': 0.08588614768791838, 'ship': 0.18384638625743577, 'helicopter': 0.06590909090909092},
'mmAP': 0.35923286694026607,
'0.7': {'ground-track-field': 0.4385066163040262, 'harbor': 0.2004849369462918, 'bridge': 0.13189991198289955, 'small-vehicle': 0.41173024457583235, 'plane': 0.7905792123899915, 'baseball-diamond': 0.33846255142519494, 'tennis-court': 0.9031235090086663, 'roundabout': 0.45296468077000096, 'storage-tank': 0.6792869554877644, 'swimming-pool': 0.1969023557455042, 'mAP': 0.4448856961613535, 'soccer-ball-field': 0.5147552299156577, 'basketball-court': 0.47906270045099153, 'large-vehicle': 0.3334752568068329, 'ship': 0.5709906745500424, 'helicopter': 0.23106060606060608},
'0.9': {'ground-track-field': 0.013986013986013986, 'harbor': 0.002932551319648094, 'bridge': 0.000282326369282891, 'small-vehicle': 0.0031978072179077205, 'plane': 0.12144979203802733, 'baseball-diamond': 0.09090909090909091, 'tennis-court': 0.3105592596206337, 'roundabout': 0.09090909090909091, 'storage-tank': 0.043532372020744114, 'swimming-pool': 0.00029231218941829873, 'mAP': 0.05292676216204492, 'soccer-ball-field': 0.05524475524475524, 'basketball-court': 0.045454545454545456, 'large-vehicle': 0.006060606060606061, 'ship': 0.009090909090909092, 'helicopter': 0.0},
'0.65': {'ground-track-field': 0.5256384950288536, 'harbor': 0.2916501930015581, 'bridge': 0.17809220559814648, 'small-vehicle': 0.5129586251041002, 'plane': 0.8894034686906369, 'baseball-diamond': 0.5249010996303538, 'tennis-court': 0.9050013758244457, 'roundabout': 0.504625741843787, 'storage-tank': 0.7537275931713616, 'swimming-pool': 0.2889168538278225, 'mAP': 0.5213593647460195, 'soccer-ball-field': 0.5539343130129118, 'basketball-court': 0.5139638068449094, 'large-vehicle': 0.4321755180088217, 'ship': 0.6335125302514466, 'helicopter': 0.3118886513511373},
'0.5': {'ground-track-field': 0.5817047190853409, 'harbor': 0.5423160296407179, 'bridge': 0.37985530785380944, 'small-vehicle': 0.6212558927508246, 'plane': 0.8991382954230245, 'baseball-diamond': 0.6884909042118417, 'tennis-court': 0.9074714532809276, 'roundabout': 0.6247024980791215, 'storage-tank': 0.7908352165588822, 'swimming-pool': 0.5101446981453137, 'mAP': 0.6433669597686625, 'soccer-ball-field': 0.709771501950316, 'basketball-court': 0.5437748871261118, 'large-vehicle': 0.6161368250574863, 'ship': 0.8084240148818748, 'helicopter': 0.4264821524843431},
'0.55': {'ground-track-field': 0.575700748371701, 'harbor': 0.48360728773857997, 'bridge': 0.32298317197853993, 'small-vehicle': 0.6060592932618177, 'plane': 0.8978626322707085, 'baseball-diamond': 0.657004331905233, 'tennis-court': 0.907337369076047, 'roundabout': 0.6011977619793185, 'storage-tank': 0.7885043330695543, 'swimming-pool': 0.48472692462266914, 'mAP': 0.6140150681924789, 'soccer-ball-field': 0.6472686724945429, 'basketball-court': 0.5309924718578253, 'large-vehicle': 0.5552623519506533, 'ship': 0.750600756135258, 'helicopter': 0.40111791617473436},
'0.95': {'ground-track-field': 0.0, 'harbor': 0.0, 'bridge': 0.0, 'small-vehicle': 0.00010078613182826043, 'plane': 0.004102785575469661, 'baseball-diamond': 0.0, 'tennis-court': 0.09090909090909091, 'roundabout': 0.0016835016835016834, 'storage-tank': 0.003621876131836291, 'swimming-pool': 0.0, 'mAP': 0.007933510175509946, 'soccer-ball-field': 0.018181818181818184, 'basketball-court': 0.0, 'large-vehicle': 0.00025826446280991736, 'ship': 0.00014452955629426219, 'helicopter': 0.0},
'0.85': {'ground-track-field': 0.12179691653375865, 'harbor': 0.00818181818181818, 'bridge': 0.011363636363636364, 'small-vehicle': 0.020008904011782284, 'plane': 0.3041595005123823, 'baseball-diamond': 0.10876623376623376, 'tennis-court': 0.6415239979360767, 'roundabout': 0.1266637317484775, 'storage-tank': 0.21079632046855917, 'swimming-pool': 0.004329004329004329, 'mAP': 0.1360229133672777, 'soccer-ball-field': 0.17866004962779156, 'basketball-court': 0.18620689655172412, 'large-vehicle': 0.02561482058270067, 'ship': 0.07928485690820646, 'helicopter': 0.012987012987012986},
'0.75': {'ground-track-field': 0.38324233567107485, 'harbor': 0.11957411957411958, 'bridge': 0.10577255444175597, 'small-vehicle': 0.2773328982910034, 'plane': 0.6717961393802804, 'baseball-diamond': 0.18744781108289382, 'tennis-court': 0.80974614279133, 'roundabout': 0.3273415371813541, 'storage-tank': 0.5539919596357566, 'swimming-pool': 0.0639939770374553, 'mAP': 0.3408238746009085, 'soccer-ball-field': 0.4580894506562955, 'basketball-court': 0.42804302074314954, 'large-vehicle': 0.2186913819763849, 'ship': 0.3686584269144099, 'helicopter': 0.13863636363636364}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_G_2x_20200929'
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


