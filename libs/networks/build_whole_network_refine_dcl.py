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
                                     num_outputs=cfgs.CLASS_NUM * self.num_anchors_per_location,
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
                                      num_outputs=4 * self.num_anchors_per_location,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                      biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                      scope=scope_list[3],
                                      activation_fn=None,
                                      reuse=reuse_flag)

        rpn_angle_cls = slim.conv2d(rpn_conv2d_3x3,
                                    num_outputs=self.coding_len * self.num_anchors_per_location,
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

        refine_box_pred_list, refine_cls_score_list, refine_cls_prob_list, refine_angle_cls_list = self.refine_net(feature_pyramid, 'refine_net')

        refine_box_pred = tf.concat(refine_box_pred_list, axis=0)
        refine_cls_score = tf.concat(refine_cls_score_list, axis=0)
        refine_cls_prob = tf.concat(refine_cls_prob_list, axis=0)
        refine_angle_cls = tf.concat(refine_angle_cls_list, axis=0)

        if cfgs.METHOD == 'H':
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            h = anchors[:, 2] - anchors[:, 0] + 1
            w = anchors[:, 3] - anchors[:, 1] + 1
            theta = -90 * tf.ones_like(x_c)
            anchors = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
        # if cfgs.ANGLE_RANGE == 180:
        #     anchors = tf.py_func(coordinate_present_convert,
        #                          inp=[anchors, -1],
        #                          Tout=[tf.float32])
        #     anchors = tf.reshape(anchors, [-1, 5])

        refine_boxes = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=rpn_box_pred)

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
                                                                        decimal_weight=True)

                self.losses_dict['refine_cls_loss'] = refine_cls_loss * cfgs.CLS_WEIGHT
                self.losses_dict['refine_reg_loss'] = refine_reg_loss * cfgs.REG_WEIGHT
                self.losses_dict['angle_cls_loss'] = angle_cls_loss * cfgs.ANGLE_WEIGHT

        with tf.variable_scope('postprocess_detctions'):
            scores, category, boxes_angle = postprocess_detctions(refine_bbox_pred=refine_box_pred,
                                                                  refine_cls_prob=refine_cls_prob,
                                                                  refine_angle_prob=tf.sigmoid(refine_angle_cls),
                                                                  refine_boxes=refine_boxes,
                                                                  is_training=self.is_training,
                                                                  gpu_id=gpu_id)
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

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients
