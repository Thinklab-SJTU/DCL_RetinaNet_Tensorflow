# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
BCL + OMEGA = 180 / 256.

{'0.85': {'bridge': 0.0303030303030303, 'mAP': 0.11042327853710288, 'ground-track-field': 0.11363636363636365, 'soccer-ball-field': 0.1567070480844277, 'swimming-pool': 0.011363636363636364, 'tennis-court': 0.5504608138870233, 'harbor': 0.012987012987012986, 'ship': 0.03956099987563736, 'small-vehicle': 0.012396694214876032, 'large-vehicle': 0.007086454747924681, 'baseball-diamond': 0.09242424242424241, 'plane': 0.2836419880709763, 'roundabout': 0.07095959595959596, 'storage-tank': 0.19114361155138287, 'helicopter': 0.011363636363636364, 'basketball-court': 0.07231404958677687},
'0.6': {'bridge': 0.2685736979891338, 'mAP': 0.581720066252559, 'ground-track-field': 0.545786552311238, 'soccer-ball-field': 0.61794796671745, 'swimming-pool': 0.39745522895695573, 'tennis-court': 0.9072994977020092, 'harbor': 0.383582757412755, 'ship': 0.7236867746336149, 'small-vehicle': 0.5420458050353067, 'large-vehicle': 0.5086492312576099, 'baseball-diamond': 0.6395996329208662, 'plane': 0.8906440344951493, 'roundabout': 0.5969351776744959, 'storage-tank': 0.7831898676578961, 'helicopter': 0.39853081566918325, 'basketball-court': 0.5218739533547208},
'0.95': {'bridge': 0.0, 'mAP': 0.001705589419215808, 'ground-track-field': 0.0, 'soccer-ball-field': 0.002840909090909091, 'swimming-pool': 0.0, 'tennis-court': 0.0136986301369863, 'harbor': 0.0, 'ship': 0.0005411255411255411, 'small-vehicle': 9.671179883945841e-05, 'large-vehicle': 0.00010809642200843151, 'baseball-diamond': 0.0, 'plane': 0.0036363636363636364, 'roundabout': 0.0, 'storage-tank': 0.004662004662004662, 'helicopter': 0.0, 'basketball-court': 0.0},
'0.5': {'bridge': 0.3767102528180646, 'mAP': 0.6497933707578156, 'ground-track-field': 0.5794795869268998, 'soccer-ball-field': 0.6669475279421138, 'swimming-pool': 0.5142202102115482, 'tennis-court': 0.9082329248805103, 'harbor': 0.5379201045559816, 'ship': 0.8042007817565741, 'small-vehicle': 0.607645770261796, 'large-vehicle': 0.6067023262767249, 'baseball-diamond': 0.6992478341372819, 'plane': 0.896332087727091, 'roundabout': 0.6815619279372005, 'storage-tank': 0.8324058254806966, 'helicopter': 0.4800664294985077, 'basketball-court': 0.555226970956243},
'0.9': {'bridge': 0.0, 'mAP': 0.03636275418803255, 'ground-track-field': 0.005681818181818182, 'soccer-ball-field': 0.04672036823935559, 'swimming-pool': 0.00020661157024793388, 'tennis-court': 0.23029905335904233, 'harbor': 0.0006226650062266501, 'ship': 0.0059547439460103215, 'small-vehicle': 0.0032467532467532465, 'large-vehicle': 0.0011086474501108647, 'baseball-diamond': 0.045454545454545456, 'plane': 0.11218930881852231, 'roundabout': 0.024793388429752063, 'storage-tank': 0.053847203584506564, 'helicopter': 0.003952569169960474, 'basketball-court': 0.011363636363636364},
'0.8': {'bridge': 0.09090909090909091, 'mAP': 0.23393775245965284, 'ground-track-field': 0.22106065465207814, 'soccer-ball-field': 0.3234015890790807, 'swimming-pool': 0.025398945772257923, 'tennis-court': 0.7739762892113322, 'harbor': 0.09090909090909091, 'ship': 0.15358788569894602, 'small-vehicle': 0.1282595010377494, 'large-vehicle': 0.05450480344985046, 'baseball-diamond': 0.23086124401913877, 'plane': 0.5165175493858523, 'roundabout': 0.200088640948856, 'storage-tank': 0.3921729302692105, 'helicopter': 0.058190601668862536, 'basketball-court': 0.24922746988339622},
'0.55': {'bridge': 0.308348341483406, 'mAP': 0.6123052663356329, 'ground-track-field': 0.5733262191221598, 'soccer-ball-field': 0.6202297537169527, 'swimming-pool': 0.4862684328841703, 'tennis-court': 0.9080974418836264, 'harbor': 0.4446039221933513, 'ship': 0.7430821770108702, 'small-vehicle': 0.5844947208680713, 'large-vehicle': 0.5605989910565321, 'baseball-diamond': 0.6595536040023213, 'plane': 0.8943758304694862, 'roundabout': 0.631431181032473, 'storage-tank': 0.8125924822339063, 'helicopter': 0.435701943722444, 'basketball-court': 0.5218739533547208},
'0.75': {'bridge': 0.09090909090909091, 'mAP': 0.3395201862640761, 'ground-track-field': 0.31014551624845743, 'soccer-ball-field': 0.48090409031058523, 'swimming-pool': 0.08402539810522072, 'tennis-court': 0.8079975808066675, 'harbor': 0.11818181818181818, 'ship': 0.3402569947584336, 'small-vehicle': 0.21904798741546247, 'large-vehicle': 0.19823145551985333, 'baseball-diamond': 0.3342251733229177, 'plane': 0.6637881212738586, 'roundabout': 0.34933484322901587, 'storage-tank': 0.566958967262493, 'helicopter': 0.10617956241684069, 'basketball-court': 0.42261619420042573},
'mmAP': 0.3540406393428824,
'0.65': {'bridge': 0.18217016788842594, 'mAP': 0.5261925845145015, 'ground-track-field': 0.5367163048614595, 'soccer-ball-field': 0.606243847858825, 'swimming-pool': 0.29212409725985683, 'tennis-court': 0.9054787104147187, 'harbor': 0.27398119971891494, 'ship': 0.6279059242394054, 'small-vehicle': 0.4721737969804415, 'large-vehicle': 0.428321784931233, 'baseball-diamond': 0.5726873523786289, 'plane': 0.8001968128954269, 'roundabout': 0.544576664314034, 'storage-tank': 0.7613307289961422, 'helicopter': 0.36927024540802067, 'basketball-court': 0.51971112957199},
'0.7': {'bridge': 0.13600129919094353, 'mAP': 0.44844554470023473, 'ground-track-field': 0.4598631987257531, 'soccer-ball-field': 0.5402438038612574, 'swimming-pool': 0.18612240651970746, 'tennis-court': 0.9023152099415884, 'harbor': 0.17244993670993633, 'ship': 0.5131022062219464, 'small-vehicle': 0.36670878844480825, 'large-vehicle': 0.3269083013545808, 'baseball-diamond': 0.44789832132081675, 'plane': 0.7825799110974935, 'roundabout': 0.4788089650564199, 'storage-tank': 0.676715123438249, 'helicopter': 0.2557164869029276, 'basketball-court': 0.4812492117170927}}


"""

# ------------------------------------------------
VERSION = 'RetinaNet_DOTA_DCL_B_2x_20200914'
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


