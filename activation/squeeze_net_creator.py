from keras.models import Model
from keras.layers import Input, Dropout, Activation, Flatten, advanced_activations, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras import initializations

from math import sqrt


def my_uniform(shape, name=None):
    scale = 1/sqrt(shape[2])
    return initializations.uniform(shape, scale=scale, name=name)

def lecun_uniform(shape, name=None):
    (fan_in, fan_out) = initializations.get_fans(shape, 'tf')
    scale = 1/sqrt(fan_in)
    return initializations.uniform(shape, scale=scale, name=name)


def xavier_half(shape, name=None):
    (fan_in, fan_out) = initializations.get_fans(shape, 'tf')
    fan_avg = (fan_in * fan_out)/2
    scale = sqrt(3/fan_in)
    return initializations.uniform(shape, scale=scale, name=name)


def get_activation_func(activation_func, name, net):
    if activation_func.__name__ == 'Activation':
        return activation_func('relu', name=name)(net)
    elif activation_func.__name__ == 'PReLU' or activation_func.__name__ == 'SReLU':
        return activation_func(name=name, shared_axes=[0, 1])(net)
    else:
        return activation_func(name=name)(net)


def SqueezeNet(neurons, shrinkage, reps, input_shape, nClasses, activation_func=advanced_activations.LeakyReLU, dropout_fac=0.25):

    weight_filler = lecun_uniform

    # Definition
    input_img = Input(shape=input_shape)
    conv1 = Convolution2D(neurons[0], 3, 3, border_mode='same',
                          name='conv1',
                          init=weight_filler)(input_img)
    elu1 = get_activation_func(activation_func, 'conv1_elu', conv1)
    current_net = MaxPooling2D(pool_size=(2, 2), name='conv1_MaxPool')(elu1)

    for l in range(1, len(neurons)):
        s_neurons = int(neurons[l-1]*shrinkage)
        e_neurons = neurons[l]

        for r in range(reps):
            # Squeeze
            fire_squeeze = Convolution2D(s_neurons, 1, 1,  border_mode='same',
                                         name=('fire%d_%d_squeeze' % (l, r)),
                                         init=weight_filler)(current_net)
            fire_squeeze_elu = get_activation_func(activation_func, 'fire%d_%d_squeeze_elu' % (l, r), fire_squeeze)
            # Expand 1x1
            fire_expand1 = Convolution2D(e_neurons, 1, 1, border_mode='same',
                                         name=('fire%d_%d_expand1' % (l, r)),
                                         init=weight_filler)(fire_squeeze_elu)
            fire_expand1_elu = get_activation_func(activation_func, 'fire%d_%d_expand1_elu' % (l, r), fire_expand1)
            # Expand 3x3
            fire_expand2 = Convolution2D(e_neurons, 3, 3, border_mode='same',
                                         name=('fire%d_%d_expand2' % (l, r)),
                                         init=weight_filler)(fire_squeeze_elu)
            fire_expand2_elu = get_activation_func(activation_func, 'fire%d_%d_expand2_elu' % (l, r), fire_expand2)
            # Merge
            current_net = merge([fire_expand1_elu, fire_expand2_elu],
                                name=('fire%d_%d_merge' % (l, r)),
                                mode='concat', concat_axis=3)

        # Pool
        current_net = MaxPooling2D(pool_size=(2, 2), name=('fire%d_MaxPool' % l))(current_net)
        # Dropout
        if dropout_fac!=0.0 and l!=(len(neurons)-1):
            current_net = Dropout(dropout_fac, name='fire%d_%d_Dropout' % (l, r))(current_net)

    # Classification layer
    current_net = Dropout(0.5, name='final_Dropout')(current_net)
    final_conv = Convolution2D(nClasses, 1, 1, border_mode='same', name='final_conv', init=weight_filler)(current_net)
    avgpool = AveragePooling2D((2, 2), name='avgpool')(final_conv)
    flatten = Flatten(name='flatten')(avgpool)
    softmax = Activation("softmax", name='softmax')(flatten)

    return Model(input=input_img, output=softmax)