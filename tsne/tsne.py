import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append("../")

from libs.configs import cfgs


LOG_DIR = './dcl_log/{}'.format(cfgs.VERSION)
SPRITE_FILE = 'dota_sprite.jpg'
META_FIEL = "dcl_meta.tsv"
TENSOR_NAME = "FINAL_LOGITS"


# 生成可视化最终输出层向量所需要的日志文件
def visualisation(final_result):
    # 使用一个新的变量来保存最终输出层向量的结果，因为embedding是通过Tensorflow中变量完成的，所以PROJECTOR可视化的都是TensorFlow中的变哇。
    # 所以这里需要新定义一个变量来保存输出层向量的取值
    y = tf.Variable(final_result, name=TENSOR_NAME)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # 通过project.ProjectorConfig类来帮助生成日志文件
    config = projector.ProjectorConfig()
    # 增加一个需要可视化的bedding结果
    embedding = config.embeddings.add()
    # 指定这个embedding结果所对应的Tensorflow变量名称
    embedding.tensor_name = y.name

    # Specify where you find the metadata
    # 指定embedding结果所对应的原始数据信息。比如这里指定的就是每一张MNIST测试图片对应的真实类别。在单词向量中可以是单词ID对应的单词。
    # 这个文件是可选的，如果没有指定那么向量就没有标签。
    embedding.metadata_path = META_FIEL

    # Specify where you find the sprite (we will create this later)
    # 指定sprite 图像。这个也是可选的，如果没有提供sprite 图像，那么可视化的结果
    # 每一个点就是一个小困点，而不是具体的图片。
    # embedding.sprite.image_path = SPRITE_FILE
    # 在提供sprite图像时，通过single_image_dim可以指定单张图片的大小。
    # 这将用于从sprite图像中截取正确的原始图片。
    # embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    # 将PROJECTOR所需要的内容写入日志文件。
    projector.visualize_embeddings(summary_writer, config)

    # 生成会话，初始化新声明的变量并将需要的日志信息写入文件。
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"))

    summary_writer.close()


# 主函数先调用模型训练的过程，再使用训练好的模型来处理MNIST测试数据，
# 最后将得到的输出层矩阵输出到PROJECTOR需要的日志文件中。
def main(argv=None):

    final_result = np.load(os.path.join(LOG_DIR, "final_logits.npy")) # [:25000, :]
    print(final_result.shape)
    final_result = tf.constant(final_result, tf.float32)
    visualisation(final_result)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    main()