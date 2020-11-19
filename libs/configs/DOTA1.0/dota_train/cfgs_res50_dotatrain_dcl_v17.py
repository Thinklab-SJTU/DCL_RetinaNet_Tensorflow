# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 256. + redundancy (1)
FLOPs: 874390815;    Trainable params: 33390156

{'0.65': {'small-vehicle': 0.4389060130382313, 'swimming-pool': 0.3014806008648077, 'tennis-court': 0.8995578122029192, 'soccer-ball-field': 0.5775430656584999, 'roundabout': 0.47218981280319566, 'bridge': 0.19606842819222756, 'storage-tank': 0.7559191709487985, 'helicopter': 0.3477173276798481, 'plane': 0.8801452488025657, 'ship': 0.6185183537975522, 'mAP': 0.5232251724228066, 'ground-track-field': 0.5728289722009468, 'harbor': 0.26199278897099854, 'large-vehicle': 0.3987478961402299, 'basketball-court': 0.5686422929568493, 'baseball-diamond': 0.5581198020844299}, '0.7': {'small-vehicle': 0.31890761950456353, 'swimming-pool': 0.18124371406808784, 'tennis-court': 0.8126552548355951, 'soccer-ball-field': 0.491300236981062, 'roundabout': 0.3953012855291076, 'bridge': 0.13948094563352972, 'storage-tank': 0.6719208844254987, 'helicopter': 0.2532414307004471, 'plane': 0.7773927947026521, 'ship': 0.49460942383216977, 'mAP': 0.4296493353255463, 'ground-track-field': 0.502404192900954, 'harbor': 0.16824285602461025, 'large-vehicle': 0.26976901837559597, 'basketball-court': 0.5618458789883042, 'baseball-diamond': 0.4064244933810151}, '0.85': {'small-vehicle': 0.018181818181818184, 'swimming-pool': 0.005145797598627788, 'tennis-court': 0.541270786016109, 'soccer-ball-field': 0.18181818181818182, 'roundabout': 0.048484848484848485, 'bridge': 0.006993006993006993, 'storage-tank': 0.17356100380762027, 'helicopter': 0.03409090909090909, 'plane': 0.22913343965975547, 'ship': 0.03006790893429401, 'mAP': 0.10573371458486845, 'ground-track-field': 0.08410306271268839, 'harbor': 0.0055248618784530384, 'large-vehicle': 0.006954102920723226, 'basketball-court': 0.18599878382487078, 'baseball-diamond': 0.0346772068511199}, '0.9': {'small-vehicle': 0.0053475935828877, 'swimming-pool': 0.0003551136363636364, 'tennis-court': 0.23187114760062405, 'soccer-ball-field': 0.10330578512396695, 'roundabout': 0.0303030303030303, 'bridge': 0.0, 'storage-tank': 0.05967830478700044, 'helicopter': 0.018181818181818184, 'plane': 0.05303874600165, 'ship': 0.008264462809917356, 'mAP': 0.04191156984582267, 'ground-track-field': 0.01048951048951049, 'harbor': 0.0023923444976076554, 'large-vehicle': 0.001549586776859504, 'basketball-court': 0.09090909090909091, 'baseball-diamond': 0.012987012987012986}, 'mmAP': 0.34883082800778265, '0.95': {'small-vehicle': 2.4596615505706414e-05, 'swimming-pool': 0.0, 'tennis-court': 0.0067209055535903785, 'soccer-ball-field': 0.0, 'roundabout': 0.0, 'bridge': 0.0, 'storage-tank': 0.03636363636363637, 'helicopter': 0.0, 'plane': 0.0034965034965034965, 'ship': 0.0001295001295001295, 'mAP': 0.0035105858593996167, 'ground-track-field': 0.0034965034965034965, 'harbor': 0.0023923444976076554, 'large-vehicle': 3.479773814702044e-05, 'basketball-court': 0.0, 'baseball-diamond': 0.0}, '0.5': {'small-vehicle': 0.5819545980731685, 'swimming-pool': 0.4887434412704065, 'tennis-court': 0.905897217012772, 'soccer-ball-field': 0.6707845658691137, 'roundabout': 0.6051984981397238, 'bridge': 0.37631505217547934, 'storage-tank': 0.7903618999327211, 'helicopter': 0.46555426618415613, 'plane': 0.8988561446113252, 'ship': 0.811045305186299, 'mAP': 0.6427678690632103, 'ground-track-field': 0.6317561786237651, 'harbor': 0.5294656999041398, 'large-vehicle': 0.5717676421188909, 'basketball-court': 0.622477194100363, 'baseball-diamond': 0.6913403327458295}, '0.6': {'small-vehicle': 0.5186685353634506, 'swimming-pool': 0.40138972332684164, 'tennis-court': 0.9013992316630886, 'soccer-ball-field': 0.6512416244218859, 'roundabout': 0.5381706878518695, 'bridge': 0.27525833051762366, 'storage-tank': 0.7768661432037949, 'helicopter': 0.39436177188286475, 'plane': 0.8901212941954775, 'ship': 0.7262795921809239, 'mAP': 0.5789011172463077, 'ground-track-field': 0.5980277370667486, 'harbor': 0.3549015225820812, 'large-vehicle': 0.4537384205368441, 'basketball-court': 0.5754681425458568, 'baseball-diamond': 0.6276240013552633}, '0.8': {'small-vehicle': 0.060970952027251474, 'swimming-pool': 0.017152658662092625, 'tennis-court': 0.7760072829900582, 'soccer-ball-field': 0.3183734578991496, 'roundabout': 0.1803230383062316, 'bridge': 0.021141649048625793, 'storage-tank': 0.35964080602042603, 'helicopter': 0.13131313131313133, 'plane': 0.4832674785270371, 'ship': 0.13926677768184714, 'mAP': 0.2190740815872853, 'ground-track-field': 0.249042944894666, 'harbor': 0.0303030303030303, 'large-vehicle': 0.042262666910554234, 'basketball-court': 0.36957529989466353, 'baseball-diamond': 0.10747004933051445}, '0.75': {'small-vehicle': 0.17698556618369615, 'swimming-pool': 0.08856749311294765, 'tennis-court': 0.8069803495312912, 'soccer-ball-field': 0.43445614100694313, 'roundabout': 0.31686927138593873, 'bridge': 0.04317794641474369, 'storage-tank': 0.5552340517009271, 'helicopter': 0.17526223776223776, 'plane': 0.649818326787998, 'ship': 0.33241001384962987, 'mAP': 0.3313457482203253, 'ground-track-field': 0.4118323875899634, 'harbor': 0.08354573775134522, 'large-vehicle': 0.14519104973169059, 'basketball-court': 0.484528233481374, 'baseball-diamond': 0.2653274170141529}, '0.55': {'small-vehicle': 0.5646119472118478, 'swimming-pool': 0.46402824080899463, 'tennis-court': 0.9030532938799415, 'soccer-ball-field': 0.6663032454165213, 'roundabout': 0.5959944046150942, 'bridge': 0.33060734343779297, 'storage-tank': 0.7862281233390838, 'helicopter': 0.40409818035549033, 'plane': 0.893821944313363, 'ship': 0.7467848085688038, 'mAP': 0.6121890859222537, 'ground-track-field': 0.6150874845759161, 'harbor': 0.4327140178901831, 'large-vehicle': 0.5400288879984534, 'basketball-court': 0.5874149696080971, 'baseball-diamond': 0.6520593968142234}}
"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20201010'
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
SAVE_WEIGHTS_INTE = 27000 * 2

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


