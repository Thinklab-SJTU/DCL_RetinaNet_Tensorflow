# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
sys.path.append("../")

from libs.configs import cfgs
from libs.networks import build_whole_network_r3det_dcl
from libs.val_libs import voc_eval, voc_eval_r
from libs.box_utils import draw_box_in_img
from libs.box_utils.coordinate_convert import forward_convert, backward_convert
from help_utils import tools
from libs.box_utils import nms_rotate
from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms


def eval_with_plac(img_dir, det_net, num_imgs, image_ext, draw_imgs=False):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)

    img_batch = tf.expand_dims(img_batch, axis=0)

    detection_scores, detection_category, detection_boxes_angle = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch_h=None,
        gtboxes_batch_r=None,
        gt_encode_label=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        all_boxes_r = []
        imgs = os.listdir(img_dir)
        pbar = tqdm(imgs)
        for a_img_name in pbar:
            a_img_name = a_img_name.split(image_ext)[0]

            raw_img = cv2.imread(os.path.join(img_dir,
                                              a_img_name + image_ext))
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            det_boxes_r_all, det_scores_r_all, det_category_r_all = [], [], []

            img_short_side_len_list = cfgs.IMG_SHORT_SIDE_LEN if isinstance(cfgs.IMG_SHORT_SIDE_LEN, list) else [
                cfgs.IMG_SHORT_SIDE_LEN]
            img_short_side_len_list = [img_short_side_len_list[0]] if not args.multi_scale else img_short_side_len_list

            for short_size in img_short_side_len_list:
                max_len = cfgs.IMG_MAX_LENGTH
                if raw_h < raw_w:
                    new_h, new_w = short_size, min(int(short_size * float(raw_w) / raw_h), max_len)
                else:
                    new_h, new_w = min(int(short_size * float(raw_h) / raw_w), max_len), short_size
                img_resize = cv2.resize(raw_img, (new_w, new_h))

                resized_img, detected_boxes, detected_scores, detected_categories = \
                    sess.run(
                        [img_batch, detection_boxes_angle, detection_scores, detection_category],
                        feed_dict={img_plac: img_resize[:, :, ::-1]}
                    )

                if detected_boxes.shape[0] == 0:
                    continue
                resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                detected_boxes = forward_convert(detected_boxes, False)
                detected_boxes[:, 0::2] *= (raw_w / resized_w)
                detected_boxes[:, 1::2] *= (raw_h / resized_h)

                det_boxes_r_all.extend(detected_boxes)
                det_scores_r_all.extend(detected_scores)
                det_category_r_all.extend(detected_categories)
            det_boxes_r_all = np.array(det_boxes_r_all)
            det_scores_r_all = np.array(det_scores_r_all)
            det_category_r_all = np.array(det_category_r_all)

            box_res_rotate_ = []
            label_res_rotate_ = []
            score_res_rotate_ = []

            if det_scores_r_all.shape[0] != 0:
                for sub_class in range(1, cfgs.CLASS_NUM + 1):
                    index = np.where(det_category_r_all == sub_class)[0]
                    if len(index) == 0:
                        continue
                    tmp_boxes_r = det_boxes_r_all[index]
                    tmp_label_r = det_category_r_all[index]
                    tmp_score_r = det_scores_r_all[index]

                    tmp_boxes_r_ = backward_convert(tmp_boxes_r, False)

                    try:
                        inx = nms_rotate.nms_rotate_cpu(boxes=np.array(tmp_boxes_r_),
                                                        scores=np.array(tmp_score_r),
                                                        iou_threshold=cfgs.NMS_IOU_THRESHOLD,
                                                        max_output_size=5000)
                    except:
                        tmp_boxes_r_ = np.array(tmp_boxes_r_)
                        tmp = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                        tmp[:, 0:-1] = tmp_boxes_r_
                        tmp[:, -1] = np.array(tmp_score_r)
                        # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                        jitter = np.zeros([tmp_boxes_r_.shape[0], tmp_boxes_r_.shape[1] + 1])
                        jitter[:, 0] += np.random.rand(tmp_boxes_r_.shape[0], ) / 1000
                        inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                             float(cfgs.NMS_IOU_THRESHOLD), 0)

                    box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                    score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                    label_res_rotate_.extend(np.array(tmp_label_r)[inx])

            if len(box_res_rotate_) == 0:
                all_boxes_r.append(np.array([]))
                continue

            det_boxes_r_ = np.array(box_res_rotate_)
            det_scores_r_ = np.array(score_res_rotate_)
            det_category_r_ = np.array(label_res_rotate_)

            if draw_imgs:
                detected_indices = det_scores_r_ >= cfgs.VIS_SCORE
                detected_scores = det_scores_r_[detected_indices]
                detected_boxes = det_boxes_r_[detected_indices]
                detected_categories = det_category_r_[detected_indices]

                detected_boxes = backward_convert(detected_boxes, False)

                det_detections_r = draw_box_in_img.draw_boxes_with_label_and_scores(raw_img[:, :, ::-1],
                                                                                    boxes=detected_boxes,
                                                                                    labels=detected_categories,
                                                                                    scores=detected_scores,
                                                                                    method=1,
                                                                                    in_graph=True)

                save_dir = os.path.join('test_hrsc', cfgs.VERSION, 'hrsc2016_img_vis')
                tools.mkdir(save_dir)

                cv2.imwrite(save_dir + '/{}.jpg'.format(a_img_name),
                            det_detections_r[:, :, ::-1])

            det_boxes_r_ = backward_convert(det_boxes_r_, False)

            x_c, y_c, w, h, theta = det_boxes_r_[:, 0], det_boxes_r_[:, 1], det_boxes_r_[:, 2], \
                                    det_boxes_r_[:, 3], det_boxes_r_[:, 4]

            boxes_r = np.transpose(np.stack([x_c, y_c, w, h, theta]))
            dets_r = np.hstack((det_category_r_.reshape(-1, 1),
                                det_scores_r_.reshape(-1, 1),
                                boxes_r))
            all_boxes_r.append(dets_r)

            pbar.set_description("Eval image %s" % a_img_name)

        # fw1 = open(cfgs.VERSION + '_detections_r.pkl', 'wb')
        # pickle.dump(all_boxes_r, fw1)
        return all_boxes_r


def eval(num_imgs, img_dir, image_ext, test_annotation_path, draw_imgs):

    r3det_dcl = build_whole_network_r3det_dcl.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                               is_training=False)

    all_boxes_r = eval_with_plac(img_dir=img_dir, det_net=r3det_dcl,
                                 num_imgs=num_imgs, image_ext=image_ext, draw_imgs=draw_imgs)

    # with open(cfgs.VERSION + '_detections_r.pkl', 'rb') as f2:
    #     all_boxes_r = pickle.load(f2)
    #
    #     print(len(all_boxes_r))

    imgs = os.listdir(img_dir)
    real_test_imgname_list = [i.split(image_ext)[0] for i in imgs]

    print(10 * "**")
    print('rotation eval:')
    voc_eval_r.voc_evaluate_detections(all_boxes=all_boxes_r,
                                       test_imgid_list=real_test_imgname_list,
                                       test_annotation_path=test_annotation_path)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a DCL network')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='images path',
                        default='/data/yangxue/dataset/HRSC2016/HRSC2016/Test/AllImages', type=str)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.bmp', type=str)
    parser.add_argument('--test_annotation_path', dest='test_annotation_path',
                        help='test annotate path',
                        default='/data/yangxue/dataset/HRSC2016/HRSC2016/Test/xmls', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)
    parser.add_argument('--draw_imgs', '-s', default=False,
                        action='store_true')
    parser.add_argument('--multi_scale', '-ms', default=False,
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    eval(np.inf, args.img_dir, args.image_ext, args.test_annotation_path, args.draw_imgs)

