from keras.layers import Activation
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU, SReLU
from keras.optimizers import SGD
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
#from keras.utils.visualize_util import plot

import lbp

from pickle import dump, load
from os import path

import matplotlib.pyplot as plt
import numpy as np

from squeeze_net_creator import SqueezeNet
from average_net import AverageNet


# learning rate scheduler
def lr_scheduler(p_epoch):
    alpha = p_epoch / epochs
    lrate = (1.0 - alpha) * initial_lrate + alpha * min_lrate
    print('Leanring rate for epoch %d: %f' % (p_epoch + 1, lrate))
    return lrate


def plot_results(p_results, p_net_name):
    # Compute means and std errors
    means = []
    errors = []
    legend = []
    for r in p_results:
        means.append(np.mean(p_results[r]))
        errors.append(np.std(p_results[r]))
        legend.append(r)

    # Replace "Activation" with ReLU
    legend = [a.replace('Activation', 'ReLU') for a in legend]

    # Define plot
    error_config = {'ecolor': '0.3'}
    index = np.arange(len(p_results))
    bar_width = 0.3
    plt.bar(index, means, bar_width, color='r',
            yerr=errors, error_kw=error_config)

    plt.xlabel('Function')
    plt.ylabel('Performance')
    plt.title('activation functions - %s net - Cifar 100' % p_net_name)
    plt.xticks(index + bar_width / 2.0, tuple(legend))
    plt.grid(axis='y')

    plt.savefig('plot_%s.png' % p_net_name)

    plt.close()


def add_lpb_features(p_imageset):
    # Create new matrix
    elements = p_imageset.shape[0]
    shape = p_imageset.shape[1:]

    imageset_lbp = np.zeros( (elements, shape[0], shape[1], shape[2]+1) )

    for i in range(elements):
        imageset_lbp[i, :, :, 0] = lbp.lbp(p_imageset[i, :, :, :])
        imageset_lbp[i, :, :, 1:] = p_imageset[i, :, :, :]

    return imageset_lbp

if __name__ == '__main__':
    ### MAIN ###

    # Common params
    decay = 1e-4
    momentum = 0.9
    batch_size = 256

    epochs = 40
    initial_lrate = 0.1
    min_lrate = 0.000001

    classes = 100

    runs = 4

    # Net params
    nets = {}
    #nets['tiny'] = [32, 64, 64, 128]
    nets['medium'] = [128, 256, 256, 512]
    #nets['big'] = [128, 512, 512, 1024]

    dropouts = {'tiny': 0.0,
                'medium': 0.25,
                'big': 0.4}

    # Load Cifar100 Data
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Add LBP features
    X_train = add_lpb_features(X_train)
    X_test = add_lpb_features(X_test)

    X_train = (X_train-128) / 128
    X_test = (X_test-128) / 128

    # Convert class vectors to binary class matrices.
    y_train = np_utils.to_categorical(y_train, classes)
    y_test = np_utils.to_categorical(y_test, classes)

    # Evaluate
    activations = []
    #activations.append(Activation)
    activations.append(LeakyReLU)
    #activations.append(ELU)
    #activations.append(PReLU)
    #activations.append(SReLU)


    results = {}
    hists = {}

    input_shape = X_train.shape[1:]

    for net in nets:
        net_results = {}
        net_hists = {}

        if not path.exists('activation_eval_%s_net.bin' % net):
            for act in activations:
                net_results[act.__name__] = []
                net_hists[act.__name__] = []

                for run in range(runs):
                    print('=========================================================================')
                    print('Current run for net config %s is %d out of %d for function %s' % (net, run + 1, runs, act.__name__))

                    # Create Net
                    model = SqueezeNet(nets[net], 0.5, 1, input_shape, classes, act, dropouts[net])
                    #model = AverageNet(nets[net], 1, input_shape, classes, act, dropouts[net])
                    model.summary()
                    #plot(model, to_file='%s_net.png' % net)

                    # Let's compile the model
                    sgd = SGD(lr=0.00, decay=decay, momentum=momentum, nesterov=True)
                    model.compile(loss='categorical_crossentropy',
                                  optimizer=sgd,
                                  metrics=['accuracy'])

                    # Now train the model
                    hist = model.fit(X_train, y_train,
                                     nb_epoch=epochs, batch_size=batch_size,
                                     validation_data=(X_test, y_test),
                                     shuffle=True,
                                     callbacks=[LearningRateScheduler(lr_scheduler)],
                                     verbose=2)

                    # Save maximum validation accuracy
                    net_results[act.__name__].append(max(hist.history["val_acc"]))
                    net_hists[act.__name__].append(hist)

                    print(net_results)

            results[net] = net_results
            hists[net] = net_hists

            # Dump results
            with open('activation_eval_%s_net.bin' % net, 'w+b') as f:
                dump(net_results, f)
                f.close()

        else:
            with open('activation_eval_%s_net.bin' % net, 'rb') as f:
                net_results = load(f)
                f.close()

        # Plot Net results
        plot_results(net_results, net)

        #with open('activation_hists_%s_net.bin' % net, 'w+b') as f:
        #    dump(net_hists, f)
        #    f.close()

    print(results)
