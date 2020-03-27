
import tensorflow as tf
from tensorflow_train.layers.layers import conv3d, concat_channels, avg_pool3d, dropout
from tensorflow_train.layers.resize import resize_trilinear, resize_tricubic
from tensorflow_train.networks.unet_base import UnetBase
from tensorflow_train.layers.initializers import selu_initializer, he_initializer


class UnetClassicAvgLinear3d(UnetBase):
    """
    U-Net with average pooling and linear upsampling.
    """
    def __init__(self, repeats=2, dropout_ratio=0.5, kernel_size=None, *args, **kwargs):
        super(UnetClassicAvgLinear3d, self).__init__(*args, **kwargs)
        self.repeats = repeats
        self.dropout_ratio = dropout_ratio
        self.kernel_size = kernel_size or [3] * 3

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return concat_channels([parallel_node, upsample_node], name='concat' + str(current_level), data_format=self.data_format)

    def contracting_block(self, node, current_level, is_training):
        for i in range(self.repeats):
            node = self.conv(node, current_level, str(i), is_training)
            if i < self.repeats and self.dropout_ratio > 0:
                node = dropout(node, self.dropout_ratio, 'drop' + str(current_level), is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        return node

    def expanding_block(self, node, current_level, is_training):
        for i in range(self.repeats):
            node = self.conv(node, current_level, str(i), is_training)
            if i < self.repeats and self.dropout_ratio > 0:
                node = dropout(node, self.dropout_ratio, 'drop' + str(current_level), is_training)
        return node

    def downsample(self, node, current_level, is_training):
        return avg_pool3d(node, [2, 2, 2], name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return resize_trilinear(node, factors=[2, 2, 2], name='upsample' + str(current_level), data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv3d(node,
                      self.num_filters(current_level),
                      self.kernel_size,
                      name='conv' + postfix,
                      activation=self.activation,
                      normalization=None,
                      is_training=is_training,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      padding=self.padding)


def network_u(input, is_training, num_labels, data_format='channels_first', activation='relu', padding='same', actual_network=None, *args, **kwargs):
    """
    The U-Net
    :param input: Input tensor.
    :param num_labels: Number of outputs.
    :param is_training: True, if training network.
    :param data_format: 'channels_first' or 'channels_last'
    :param actual_network: The actual u-net instance used as the local appearance network.
    :param padding: Padding parameter passed to the convolution operations.
    :param activation: The activation function. 'relu' or 'selu'
    :param args: Not used.
    :param kwargs: Passed to actual_network()
    :return: prediction
    """
    if activation == 'relu':
        kernel_initializer = he_initializer
        activation = tf.nn.relu
    else:
        kernel_initializer = selu_initializer
        activation = tf.nn.selu
    local_kernel_initializer = tf.initializers.truncated_normal(stddev=0.001)
    local_activation = None
    with tf.variable_scope('local'):
        unet = actual_network(data_format=data_format, kernel_initializer=kernel_initializer, activation=activation, padding=padding, **kwargs)
        prediction = unet(input, is_training=is_training)
        prediction = conv3d(prediction, num_labels, [3, 3, 3], name='output', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training, data_format=data_format)
    return prediction


def spatial_configuration_net(input, num_labels, is_training, data_format='channels_first', actual_network=None, padding=None, spatial_downsample=8, *args, **kwargs):
    """
    The spatial configuration net.
    :param input: Input tensor.
    :param num_labels: Number of outputs.
    :param is_training: True, if training network.
    :param data_format: 'channels_first' or 'channels_last'
    :param actual_network: The actual u-net instance used as the local appearance network.
    :param padding: Padding parameter passed to the convolution operations.
    :param spatial_downsample: Downsamping factor for the spatial configuration stage.
    :param args: Not used.
    :param kwargs: Not used.
    :return: heatmaps, local_heatmaps, spatial_heatmaps
    """
    num_filters_base = 64
    activation = lambda x, name: tf.nn.leaky_relu(x, name=name, alpha=0.1)
    heatmap_layer_kernel_initializer = tf.truncated_normal_initializer(stddev=0.001)
    downsampling_factor = spatial_downsample
    node = conv3d(input,
                  filters=num_filters_base,
                  kernel_size=[3, 3, 3],
                  name='conv0',
                  activation=activation,
                  data_format=data_format,
                  is_training=is_training)
    scnet_local = actual_network(num_filters_base=num_filters_base,
                                 num_levels=4,
                                 double_filters_per_level=False,
                                 normalization=None,
                                 activation=activation,
                                 data_format=data_format,
                                 padding=padding)
    unet_out = scnet_local(node, is_training)
    local_heatmaps = conv3d(unet_out,
                            filters=num_labels,
                            kernel_size=[3, 3, 3],
                            name='local_heatmaps',
                            kernel_initializer=heatmap_layer_kernel_initializer,
                            activation=None,
                            data_format=data_format,
                            is_training=is_training)
    downsampled = avg_pool3d(local_heatmaps, [downsampling_factor] * 3, name='local_downsampled', data_format=data_format)
    conv = conv3d(downsampled, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv0', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv1', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_filters_base, kernel_size=[7, 7, 7], name='sconv2', activation=activation, data_format=data_format, is_training=is_training, padding=padding)
    conv = conv3d(conv, filters=num_labels, kernel_size=[7, 7, 7], name='spatial_downsampled', kernel_initializer=heatmap_layer_kernel_initializer, activation=tf.nn.tanh, data_format=data_format, is_training=is_training, padding=padding)
    spatial_heatmaps = resize_tricubic(conv, factors=[downsampling_factor] * 3, name='spatial_heatmaps', data_format=data_format)

    heatmaps = local_heatmaps * spatial_heatmaps

    return heatmaps, local_heatmaps, spatial_heatmaps
