

import os
from os import path as op
from functools import partial
from datetime import datetime as dt
import pprint
from keras.preprocessing.image import ImageDataGenerator

import ssl

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, rmsprop, SGD
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preproc

from keras.callbacks import (ModelCheckpoint, EarlyStopping, TensorBoard,
                             ReduceLROnPlateau)
from hyperopt import fmin, Trials, STATUS_OK, tpe
import yaml

from train_config_pd import (tboard_dir, ckpt_dir, data_dir,
                    model_params as MP, train_params as TP, data_flow as DF)


######################
# Params for hyperopt
######################

def get_optimizer(opt_params, lr):
    """Helper to get optimizer from text params"""
    if opt_params['opt_func'] == 'sgd':
        return SGD(lr=lr, momentum=opt_params['momentum'])
    elif opt_params['opt_func'] == 'adam':
        return Adam(lr=lr)
    elif opt_params['opt_func'] == 'rmsprop':
        return rmsprop(lr=lr)
    else:
        raise ValueError


def vgg16_net(params):
    """Train the Xception network"""
    K.clear_session()  # Remove any existing graphs
    mst_str = dt.now().strftime("%m%d_%H%M%S")

    print('\n' + '=' * 40 + '\nStarting model at {}'.format(mst_str))
    # print('Model # %s' % len(trials))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)
    ###################################
    # Set up generators
    ###################################
    train_gen = ImageDataGenerator(preprocessing_function=vgg_preproc,
                                   **DF['image_data_generator'])
    test_gen = ImageDataGenerator(preprocessing_function=vgg_preproc)

    ######################
    # Paths and Callbacks
    ######################
    ckpt_fpath = op.join(ckpt_dir, mst_str + '_L{val_loss:.2f}_E{epoch:02d}_weights.h5')
    tboard_model_dir = op.join(tboard_dir, mst_str)

    callbacks_phase1 = [TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                                    write_grads=False, embeddings_freq=0,
                                    embeddings_layer_names=['dense_preoutput', 'dense_output'])]
    callbacks_phase2 = [
        TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                    write_grads=False, embeddings_freq=0,
                    embeddings_layer_names=['dense_preoutput', 'dense_output']),
        ModelCheckpoint(ckpt_fpath, monitor='val_categorical_accuracy',
                        save_weights_only=True, save_best_only=False),
        EarlyStopping(min_delta=0.01,
                      patience=5, verbose=1),
        ReduceLROnPlateau(min_delta=0.1,
                          patience=3, verbose=1)]

    #########################
    # Construct model
    #########################
    # Get the original xception model pre-initialized weights
    ssl._create_default_https_context = ssl._create_unverified_context
    base_model = VGG16(weights='imagenet',
                          include_top=False,  # Peel off top layer
                          input_shape=TP['img_size'],
                          pooling='avg')  # Global average pooling

    x = base_model.output  # Get final layer of base XCeption model

    # Add a fully-connected layer
    x = Dense(params['dense_size'], activation=params['dense_activation'],
              kernel_initializer=params['weight_init'],
              name='dense_preoutput')(x)
    if params['dropout_rate'] > 0:
        x = Dropout(rate=params['dropout_rate'])(x)

    # Finally, add output layer
    pred = Dense(params['n_classes'],
                 activation=params['output_activation'],
                 name='dense_output')(x)

    model = Model(inputs=base_model.input, outputs=pred)

    #####################
    # Save model details
    #####################
    model_yaml = model.to_yaml()
    save_template = op.join(ckpt_dir, mst_str + '_{}.{}')
    arch_fpath = save_template.format('arch', 'yaml')
    if not op.exists(arch_fpath):
        with open(arch_fpath.format('arch', 'yaml'), 'w') as yaml_file:
            yaml_file.write(model_yaml)

    # Save params to yaml file
    params_fpath = save_template.format('params', 'yaml')
    if not op.exists(params_fpath):
        with open(params_fpath, 'w') as yaml_file:
            yaml_file.write(yaml.dump(params))
            yaml_file.write(yaml.dump(TP))
            yaml_file.write(yaml.dump(MP))
            yaml_file.write(yaml.dump(DF))

    ##########################
    # Train the new top layers
    ##########################
    # Train the top layers which we just added by setting all orig layers untrainable
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model (after setting non-trainable layers)
    model.compile(optimizer=get_optimizer(params['optimizer'],
                                          lr=params['lr_phase1']),
                  loss=params['loss'],
                  metrics=MP['metrics'])

    print('Phase 1, training near-output layer(s)')
    hist = model.fit_generator(
        train_gen.flow_from_directory(directory=op.join(data_dir, 'train'),
                                      **DF['flow_from_dir']),
        steps_per_epoch=params['steps_per_train_epo'],
        epochs=params['n_epo_phase1'],
        callbacks=callbacks_phase1,
        max_queue_size=params['max_queue_size'],
        workers=params['workers'],
        use_multiprocessing=params['use_multiprocessing'],
        class_weight=params['class_weight'],
        verbose=1)

    ###############################################
    # Train entire network to fine-tune performance
    ###############################################
    # Visualize layer names/indices to see how many layers to freeze:
    #print('Layer freeze cutoff = {}'.format(params['freeze_cutoff']))
    #for li, layer in enumerate(base_model.layers):
    #    print(li, layer.name)

    # Set all layers trainable
    for layer in model.layers:
        layer.trainable = True

    # Recompile model for second round of training
    model.compile(optimizer=get_optimizer(params['optimizer'],
                                          params['lr_phase2']),
                  loss=params['loss'],
                  metrics=MP['metrics'])

    print('/nPhase 2, training from layer {} on.'.format(params['freeze_cutoff']))
    test_iter = test_gen.flow_from_directory(
        directory=op.join(data_dir, 'test'), shuffle=False,  # Helps maintain consistency in testing phase
        **DF['flow_from_dir'])
    test_iter.reset()  # Reset for each model so it's consistent; ideally should reset every epoch

    hist = model.fit_generator(
        train_gen.flow_from_directory(directory=op.join(data_dir, 'train'),
                                      **DF['flow_from_dir']),
        steps_per_epoch=params['steps_per_train_epo'],
        epochs=params['n_epo_phase2'],
        max_queue_size=params['max_queue_size'],
        workers=params['workers'],
        use_multiprocessing=params['use_multiprocessing'],
        validation_data=test_iter,
        validation_steps=params['steps_per_test_epo'],
        callbacks=callbacks_phase2,
        class_weight=params['class_weight'],
        verbose=1)

    # Return best of last validation accuracies
    check_ind = -1 * (TP['early_stopping_patience'] + 1)
    result_dict = dict(loss=np.min(hist.history['val_loss'][check_ind:]),
                       status=STATUS_OK)

    return result_dict