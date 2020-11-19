# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet, resnet_gluoncv, mobilenet_v2, xception
from libs.box_utils import anchor_utils, generate_anchors, generate_rotate_anchors
from libs.configs import cfgs
from libs.losses import losses, losses_dcl
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.refine_proposal_opr_dcl import postprocess_detctions
from libs.detection_oprations.anchor_target_layer_without_boxweight_ import anchor_target_layer
from libs.detection_oprations.refinebox_target_layer_without_boxweight_dcl import refinebox_target_layer
from libs.box_utils import bbox_transform
from help_utils.densely_coded_label import get_code_len


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        if cfgs.METHOD == 'H':
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        else:
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS) * len(cfgs.ANCHOR_ANGLES)
        self.method = cfgs.METHOD
        self.losses_dict = {}
        self.coding_len = get_code_len(int(cfgs.ANGLE_RANGE / cfgs.OMEGA), mode=cfgs.ANGLE_MODE)

    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:

            return resnet_gluoncv.resnet_base(input_img_batch, scope_name=self.base_network_name,
                                              is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)
        else:
            raise ValueError('Sry, we only support resnet, mobilenet_v2')

    def rpn_cls_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(4):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=256,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                         scope='{}_{}'.format(scope_list[0], i),
                                         reuse=reuse_flag)

        rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=cfgs.CLASS_NUM * self.num_anchors_per_location,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=cfgs.FINAL_CONV_BIAS_INITIALIZER,
                                     scope=scope_list[2],
                                     activation_fn=None,
                                     reuse=reuse_flag)

        rpn_box_scores = tf.reshape(rpn_box_scores, [-1, cfgs.CLASS_NUM],
                                    name='rpn_{}_classification_reshape'.format(level))
        rpn_box_probs = tf.sigmoid(rpn_box_scores, name='rpn_{}_classification_sigmoid'.format(level))

        return rpn_box_scores, rpn_box_probs

    def refine_cls_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(4):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=256,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                         scope='{}_{}'.format(scope_list[0], i),
                                         reuse=reuse_flag)

        rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                     num_outputs=cfgs.CLASS_NUM,
                                     kernel_size=[3, 3],
                                     stride=1,
                                     weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                     biases_initializer=cfgs.FINAL_CONV_BIAS_INITIALIZER,
                                     scope=scope_list[2],
                                     activation_fn=None,
                                     reuse=reuse_flag)

        rpn_box_scores = tf.reshape(rpn_box_scores, [-1, cfgs.CLASS_NUM],
                                    name='refine_{}_classification_reshape'.format(level))
        rpn_box_probs = tf.sigmoid(rpn_box_scores, name='refine_{}_classification_sigmoid'.format(level))

        return rpn_box_scores, rpn_box_probs

    def rpn_reg_net(self, inputs, scope_list, reuse_flag, level):
        rpn_delta_boxes = inputs
        for i in range(4):
            rpn_delta_boxes = slim.conv2d(inputs=rpn_delta_boxes,
                                          num_outputs=256,
                                          kernel_size=[3, 3],
                                          weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                          biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                          stride=1,
                                          activation_fn=tf.nn.relu,
                                          scope='{}_{}'.format(scope_list[1], i),
                                          reuse=reuse_flag)

        rpn_delta_boxes = slim.conv2d(rpn_delta_boxes,
                                      num_outputs=5 * self.num_anchors_per_location,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                      biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                      scope=scope_list[3],
                                      activation_fn=None,
                                      reuse=reuse_flag)

        rpn_delta_boxes = tf.reshape(rpn_delta_boxes, [-1, 5],
                                     name='rpn_{}_regression_reshape'.format(level))
        return rpn_delta_boxes

    def refine_reg_net(self, inputs, scope_list, reuse_flag, level):
        rpn_conv2d_3x3 = inputs
        for i in range(4):
            rpn_conv2d_3x3 = slim.conv2d(inputs=rpn_conv2d_3x3,
                                         num_outputs=256,
                                         kernel_size=[3, 3],
                                         weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                         biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                         stride=1,
                                         activation_fn=tf.nn.relu,
                                         scope='{}_{}'.format(scope_list[1], i),
                                         reuse=reuse_flag)

        rpn_delta_boxes = slim.conv2d(rpn_conv2d_3x3,
                                      num_outputs=4,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                      biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                      scope=scope_list[3],
                                      activation_fn=None,
                                      reuse=reuse_flag)

        rpn_angle_cls = slim.conv2d(rpn_conv2d_3x3,
                                    num_outputs=self.coding_len,
                                    kernel_size=[3, 3],
                                    stride=1,
                                    weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope=scope_list[4],
                                    activation_fn=None,
                                    reuse=reuse_flag)

        rpn_delta_boxes = tf.reshape(rpn_delta_boxes, [-1, 4],
                                     name='rpn_{}_regression_reshape'.format(level))
        rpn_angle_cls = tf.reshape(rpn_angle_cls, [-1, self.coding_len],
                                   name='rpn_{}_angle_cls_reshape'.format(level))
        return rpn_delta_boxes, rpn_angle_cls

    def rpn_net(self, feature_pyramid, name):

        rpn_delta_boxes_list = []
        rpn_scores_list = []
        rpn_probs_list = []
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                for level in cfgs.LEVEL:

                    if cfgs.SHARE_NET:
                        reuse_flag = None if level == 'P3' else True
                        scope_list = ['conv2d_3x3_cls', 'conv2d_3x3_reg', 'rpn_classification', 'rpn_regression']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_cls_' + level, 'conv2d_3x3_reg_' + level,
                                      'rpn_classification_' + level, 'rpn_regression_' + level]

                    rpn_box_scores, rpn_box_probs = self.rpn_cls_net(feature_pyramid[level],
                                                                     scope_list, reuse_flag,
                                                                     level)
                    rpn_delta_boxes = self.rpn_reg_net(feature_pyramid[level], scope_list, reuse_flag, level)

                    rpn_scores_list.append(rpn_box_scores)
                    rpn_probs_list.append(rpn_box_probs)
                    rpn_delta_boxes_list.append(rpn_delta_boxes)

                # rpn_all_delta_boxes = tf.concat(rpn_delta_boxes_list, axis=0)
                # rpn_all_boxes_scores = tf.concat(rpn_scores_list, axis=0)
                # rpn_all_boxes_probs = tf.concat(rpn_probs_list, axis=0)

            return rpn_delta_boxes_list, rpn_scores_list, rpn_probs_list

    def refine_net(self, feature_pyramid, name):

        refine_delta_boxes_list = []
        refine_scores_list = []
        refine_probs_list = []
        refine_angle_cls_list = []
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                for level in cfgs.LEVEL:

                    if cfgs.SHARE_NET:
                        reuse_flag = None if level == 'P3' else True
                        scope_list = ['conv2d_3x3_cls', 'conv2d_3x3_reg', 'refine_classification',
                                      'refine_regression', 'refine_angle_cls']
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_cls_' + level, 'conv2d_3x3_reg_' + level,
                                      'refine_classification_' + level, 'refine_regression_' + level,
                                      'refine_angle_cls_' + level]

                    refine_box_scores, refine_box_probs = self.refine_cls_net(feature_pyramid[level],
                                                                              scope_list, reuse_flag,
                                                                              level)
                    refine_delta_boxes, refine_angle_cls = self.refine_reg_net(feature_pyramid[level], scope_list, reuse_flag, level)

                    refine_scores_list.append(refine_box_scores)
                    refine_probs_list.append(refine_box_probs)
                    refine_delta_boxes_list.append(refine_delta_boxes)
                    refine_angle_cls_list.append(refine_angle_cls)

            return refine_delta_boxes_list, refine_scores_list, refine_probs_list, refine_angle_cls_list

    def make_anchors(self, feature_pyramid):
        with tf.variable_scope('make_anchors'):
            anchor_list = []
            level_list = cfgs.LEVEL
            with tf.name_scope('make_anchors_all_level'):
                for level, base_anchor_size, stride in zip(level_list, cfgs.BASE_ANCHOR_SIZE_LIST, cfgs.ANCHOR_STRIDE):
                    '''
                    (level, base_anchor_size) tuple:
                    (P3, 32), (P4, 64), (P5, 128), (P6, 256), (P7, 512)
                    '''
                    featuremap_height, featuremap_width = tf.shape(feature_pyramid[level])[1], \
                                                          tf.shape(feature_pyramid[level])[2]

                    featuremap_height = tf.cast(featuremap_height, tf.float32)
                    featuremap_width = tf.cast(featuremap_width, tf.float32)

                    if self.method == 'H':
                        tmp_anchors = tf.py_func(generate_anchors.generate_anchors_pre,
                                                 inp=[featuremap_height, featuremap_width, stride,
                                                      np.array(cfgs.ANCHOR_SCALES) * stride, cfgs.ANCHOR_RATIOS, 4.0],
                                                 Tout=[tf.float32])

                        tmp_anchors = tf.reshape(tmp_anchors, [-1, 4])
                    else:
                        tmp_anchors = generate_rotate_anchors.make_anchors(base_anchor_size=base_anchor_size,
                                                                           anchor_scales=cfgs.ANCHOR_SCALES,
                                                                           anchor_ratios=cfgs.ANCHOR_RATIOS,
                                                                           anchor_angles=cfgs.ANCHOR_ANGLES,
                                                                           featuremap_height=featuremap_height,
                                                                           featuremap_width=featuremap_width,
                                                                           stride=stride)
                        tmp_anchors = tf.reshape(tmp_anchors, [-1, 5])
                    anchor_list.append(tmp_anchors)

                # all_level_anchors = tf.concat(anchor_list, axis=0)
            return anchor_list

    def add_anchor_img_smry(self, img, anchors, labels, method):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        # negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        # negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=positive_anchor,
                                                        method=method)
        # neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
        #                                                 boxes=negative_anchor)

        tf.summary.image('positive_anchor', pos_in_img)
        # tf.summary.image('negative_anchors', neg_in_img)

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h, gtboxes_batch_r, gt_encode_label, gpu_id=0):

        if self.is_training:
            gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [-1, 5])
            gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

            gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [-1, 6])
            gtboxes_batch_r = tf.cast(gtboxes_batch_r, tf.float32)

            gt_encode_label = tf.reshape(gt_encode_label, [-1, self.coding_len])
            gt_encode_label = tf.cast(gt_encode_label, tf.float32)

        # 1. build base network
        feature_pyramid = self.build_base_network(input_img_batch)

        # 2. build rpn
        rpn_box_pred_list, rpn_cls_score_list, rpn_cls_prob_list = self.rpn_net(feature_pyramid, 'rpn_net')

        # 3. generate_anchors
        anchor_list = self.make_anchors(feature_pyramid)
        rpn_box_pred = tf.concat(rpn_box_pred_list, axis=0)
        rpn_cls_score = tf.concat(rpn_cls_score_list, axis=0)
        # rpn_cls_prob = tf.concat(rpn_cls_prob_list, axis=0)
        anchors = tf.concat(anchor_list, axis=0)

        if self.is_training:
            with tf.variable_scope('build_loss'):
                labels, target_delta, anchor_states, target_boxes = tf.py_func(func=anchor_target_layer,
                                                                               inp=[gtboxes_batch_h, gtboxes_batch_r,
                                                                                    anchors],
                                                                               Tout=[tf.float32, tf.float32,
                                                                                     tf.float32, tf.float32])

                if self.method == 'H':
                    self.add_anchor_img_smry(input_img_batch, anchors, anchor_states, 0)
                else:
                    self.add_anchor_img_smry(input_img_batch, anchors, anchor_states, 1)

                cls_loss = losses.focal_loss(labels, rpn_cls_score, anchor_states)
                if cfgs.USE_IOU_FACTOR:
                    reg_loss = losses.iou_smooth_l1_loss_(target_delta, rpn_box_pred, anchor_states, target_boxes, anchors)
                else:
                    reg_loss = losses.smooth_l1_loss(target_delta, rpn_box_pred, anchor_states)

                self.losses_dict['cls_loss'] = cls_loss * cfgs.CLS_WEIGHT
                self.losses_dict['reg_loss'] = reg_loss * cfgs.REG_WEIGHT

        with tf.variable_scope('refine_feature_pyramid'):
            refine_feature_pyramid = {}
            refine_boxes_list = []

            for box_pred, cls_prob, anchor, stride, level in \
                    zip(rpn_box_pred_list, rpn_cls_prob_list, anchor_list,
                        cfgs.ANCHOR_STRIDE, cfgs.LEVEL):

                box_pred = tf.reshape(box_pred, [-1, self.num_anchors_per_location, 5])
                anchor = tf.reshape(anchor, [-1, self.num_anchors_per_location, 5 if self.method == 'R' else 4])
                cls_prob = tf.reshape(cls_prob, [-1, self.num_anchors_per_location, cfgs.CLASS_NUM])

                cls_max_prob = tf.reduce_max(cls_prob, axis=-1)
                box_pred_argmax = tf.cast(tf.reshape(tf.argmax(cls_max_prob, axis=-1), [-1, 1]), tf.int32)
                indices = tf.cast(tf.cumsum(tf.ones_like(box_pred_argmax), axis=0), tf.int32) - tf.constant(1, tf.int32)
                indices = tf.concat([indices, box_pred_argmax], axis=-1)

                box_pred_filter = tf.reshape(tf.gather_nd(box_pred, indices), [-1, 5])
                anchor_filter = tf.reshape(tf.gather_nd(anchor, indices), [-1, 5 if self.method == 'R' else 4])

                if cfgs.METHOD == 'H':
                    x_c = (anchor_filter[:, 2] + anchor_filter[:, 0]) / 2
                    y_c = (anchor_filter[:, 3] + anchor_filter[:, 1]) / 2
                    h = anchor_filter[:, 2] - anchor_filter[:, 0] + 1
                    w = anchor_filter[:, 3] - anchor_filter[:, 1] + 1
                    theta = -90 * tf.ones_like(x_c)
                    anchor_filter = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

                boxes_filter = bbox_transform.rbbox_transform_inv(boxes=anchor_filter, deltas=box_pred_filter)
                refine_boxes_list.append(boxes_filter)
                center_point = boxes_filter[:, :2] / stride

                refine_feature_pyramid[level] = self.refine_feature_op(points=center_point,
                                                                       feature_map=feature_pyramid[level],
                                                                       name=level)

        refine_box_pred_list, refine_cls_score_list, refine_cls_prob_list, refine_angle_cls_list = self.refine_net(refine_feature_pyramid, 'refine_net')

        refine_box_pred = tf.concat(refine_box_pred_list, axis=0)
        refine_cls_score = tf.concat(refine_cls_score_list, axis=0)
        refine_cls_prob = tf.concat(refine_cls_prob_list, axis=0)
        refine_angle_cls = tf.concat(refine_angle_cls_list, axis=0)
        refine_boxes = tf.concat(refine_boxes_list, axis=0)

        # 4. postprocess rpn proposals. such as: decode, clip, filter
        if self.is_training:
            with tf.variable_scope('build_refine_loss'):
                refine_labels, refine_target_delta, refine_box_states, refine_target_boxes, refine_target_encode_label = tf.py_func(
                    func=refinebox_target_layer,
                    inp=[gtboxes_batch_r, gt_encode_label, refine_boxes,
                         cfgs.REFINE_IOU_POSITIVE_THRESHOLD[0], cfgs.REFINE_IOU_NEGATIVE_THRESHOLD[0], gpu_id],
                    Tout=[tf.float32, tf.float32, tf.float32,
                          tf.float32, tf.float32])

                self.add_anchor_img_smry(input_img_batch, refine_boxes, refine_box_states, 1)

                refine_cls_loss = losses.focal_loss(refine_labels, refine_cls_score, refine_box_states)
                refine_reg_loss = losses.smooth_l1_loss(refine_target_delta, refine_box_pred, refine_box_states)
                angle_cls_loss = losses_dcl.angle_cls_period_focal_loss(refine_target_encode_label, refine_angle_cls,
                                                                        refine_box_states, refine_target_boxes,
                                                                        decimal_weight=cfgs.DATASET_NAME.startswith('DOTA'))

                self.losses_dict['refine_cls_loss'] = refine_cls_loss * cfgs.CLS_WEIGHT
                self.losses_dict['refine_reg_loss'] = refine_reg_loss * cfgs.REG_WEIGHT
                self.losses_dict['angle_cls_loss'] = angle_cls_loss * cfgs.ANGLE_WEIGHT

        with tf.variable_scope('postprocess_detctions'):
            scores, category, boxes_angle = postprocess_detctions(refine_bbox_pred=refine_box_pred,
                                                                  refine_cls_prob=refine_cls_prob,
                                                                  refine_angle_prob=tf.sigmoid(refine_angle_cls),
                                                                  refine_boxes=refine_boxes,
                                                                  is_training=self.is_training)
            scores = tf.stop_gradient(scores)
            category = tf.stop_gradient(category)
            boxes_angle = tf.stop_gradient(boxes_angle)

        if self.is_training:
            return scores, category, boxes_angle, self.losses_dict
        else:
            return scores, category, boxes_angle

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()

            # for var in model_variables:
            #     print(var.name)
            # print(20*"__++__++__")

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN/'+self.base_network_name):  # +'/block4'
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"___")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

    def refine_feature_op(self, points, feature_map, name):

        h, w = tf.cast(tf.shape(feature_map)[1], tf.int32), tf.cast(tf.shape(feature_map)[2], tf.int32)

        xmin = tf.maximum(0.0, tf.floor(points[:, 0]))
        xmin = tf.minimum(tf.cast(w - 1, tf.float32), tf.ceil(xmin))

        ymin = tf.maximum(0.0, tf.floor(points[:, 1]))
        ymin = tf.minimum(tf.cast(h - 1, tf.float32), tf.ceil(ymin))

        xmax = tf.minimum(tf.cast(w - 1, tf.float32), tf.ceil(points[:, 0]))
        xmax = tf.maximum(0.0, tf.floor(xmax))

        ymax = tf.minimum(tf.cast(h - 1, tf.float32), tf.ceil(points[:, 1]))
        ymax = tf.maximum(0.0, tf.floor(ymax))

        left_top = tf.cast(tf.transpose(tf.stack([ymin, xmin], axis=0)), tf.int32)
        right_bottom = tf.cast(tf.transpose(tf.stack([ymax, xmax], axis=0)), tf.int32)
        left_bottom = tf.cast(tf.transpose(tf.stack([ymax, xmin], axis=0)), tf.int32)
        right_top = tf.cast(tf.transpose(tf.stack([ymin, xmax], axis=0)), tf.int32)

        feature = feature_map

        left_top_feature = tf.gather_nd(tf.squeeze(feature), left_top)
        right_bottom_feature = tf.gather_nd(tf.squeeze(feature), right_bottom)
        left_bottom_feature = tf.gather_nd(tf.squeeze(feature), left_bottom)
        right_top_feature = tf.gather_nd(tf.squeeze(feature), right_top)

        refine_feature = right_bottom_feature * tf.tile(
            tf.reshape((tf.abs((points[:, 0] - xmin) * (points[:, 1] - ymin))), [-1, 1]),
            [1, cfgs.FPN_CHANNEL]) \
                         + left_top_feature * tf.tile(
            tf.reshape((tf.abs((xmax - points[:, 0]) * (ymax - points[:, 1]))), [-1, 1]),
            [1, cfgs.FPN_CHANNEL]) \
                         + right_top_feature * tf.tile(
            tf.reshape((tf.abs((points[:, 0] - xmin) * (ymax - points[:, 1]))), [-1, 1]),
            [1, cfgs.FPN_CHANNEL]) \
                         + left_bottom_feature * tf.tile(
            tf.reshape((tf.abs((xmax - points[:, 0]) * (points[:, 1] - ymin))), [-1, 1]),
            [1, cfgs.FPN_CHANNEL])

        refine_feature = tf.reshape(refine_feature, [1, tf.cast(h, tf.int32), tf.cast(w, tf.int32), cfgs.FPN_CHANNEL])

        # refine_feature = tf.reshape(refine_feature, [1, tf.cast(feature_size[1], tf.int32),
        #                                              tf.cast(feature_size[0], tf.int32), 256])

        return refine_feature + feature
