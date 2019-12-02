

"""
train_config_pd.py

List some configuration parameters for training model
"""

import os
from os import path as op
from sklearn.metrics import (precision_recall_fscore_support,
                             #f1_score, fbeta_score,
                             classification_report)
import keras
import keras.backend as K
from hyperopt import hp
from focal_losses import categorical_focal_loss
import tensorflow as tf
# from focal_losses import categorical_focal_loss_fixed
# Set directories for saving model weights and tensorboard information
data_dir = os.getcwd()

#     cloud_comp = False

ckpt_dir = op.join(os.getcwd(), "models")
tboard_dir = op.join(os.getcwd(), "tensorboard")
preds_dir = op.join(os.getcwd(), "preds")
plot_dir = op.join(os.getcwd(), "plots")
cloud_comp = False

if not op.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
if not op.isdir(tboard_dir):
    os.mkdir(tboard_dir)

class ClasswisePerformance(keras.callbacks.Callback):
    """Callback to calculate precision, recall, F1-score after each epoch"""

    def __init__(self, test_gen, gen_steps=100):
        test_gen.shuffle = False
        self.test_gen = test_gen
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.gen_steps = gen_steps

    def on_train_begin(self, logs={}):
        self.precisions = []
        self.recalls = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs={}):

        # TODO: Not efficient as this requires re-computing test data
        self.test_gen.reset()
        y_true = self.test_gen.classes
        class_labels = list(self.test_gen.class_indices.keys())

        # Leave steps=None to predict entire sequence
        y_pred_probs = self.model.predict_generator(self.test_gen,
                                                    steps=self.gen_steps)
        y_pred = np.argmax(y_pred_probs, axis=1)

        prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                              labels=class_labels)
        self.precisions.append(prec)
        self.recalls.append(recall)
        self.f1s.append(f1)

        self.test_gen.reset()
        print(classification_report(y_true, y_pred))

def focal_loss(y_true, y_pred):

    gamma = 2.0
    alpha = 4.0

#     def focal_loss_fixed(y_true, y_pred):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})

    Returns:
        [tensor] -- loss.
    """
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
#         return tf.reduce_mean(reduced_fl)
    return tf.reduce_mean(reduced_fl)

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

    return fbeta_score

def f1_score(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1.)


def f2_score(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=2.)

model_params = dict(loss=[focal_loss],
#                     loss=[focal_loss(alpha=.25, gamma=2)],
                    optimizer=[dict(opt_func='rmsprop'),
                               dict(opt_func='sgd', momentum=hp.uniform('momentum', 0.4, 0.9))],
                               # SGD as below performed notably poorer in 1st big hyperopt run

                    lr_phase1=[1e-7, 1e-3],  # learning rate for phase 1 (output layer only)
                    lr_phase2=[5e-8, 1e-4],  # learning rate for phase 2 (all layers beyond freeze_cutoff)
                    weight_init=['glorot_uniform'],
                    metrics=['categorical_accuracy'],
                    # Blocks organized in 10s, 66, 76, 86, etc.
                    freeze_cutoff=[0],  # Layer below which no training/updating occurs on weights
                    dense_size=[128, 256, 512],  # Number of nodes in 2nd to final layer
                    n_classes=5,  # Number of class choices in final layer
                    output_activation=['softmax'],
                    dense_activation=['relu', 'elu'],
                    dropout_rate=[0, 0.1, 0.25, 0.5])  # Dropout in final layer

train_params = dict(n_rand_hp_iters=1,
                    n_total_hp_iters=100,  # Total number of HyperParam experiments to run
                    n_epo_phase1=[1, 3],  # Number of epochs training only top layer
                    n_epo_phase2=20,  # Number of epochs fine tuning whole model

                    max_queue_size=128,
                    workers=8,
                    use_multiprocessing=False,
                    img_size=(46, 46, 3),
                    early_stopping_patience=10,  # Number of iters w/out val_acc increase
                    early_stopping_min_delta=0.01,
                    reduce_lr_patience=6,  # Number of iters w/out val_acc increase
                    reduce_lr_min_delta=0.1,
                    class_weight={2: 8.0, 1:5.5, 0:1.0, 3:15, 4:60},  # Based on pakistan_redux image counts
                    steps_per_train_epo=128,
                    steps_per_test_epo=293)  #586

# Define params for ImageDataGenerator and ImageDataGenerator.flow_from_directory
data_flow = dict(image_data_generator=dict(horizontal_flip=True,
                                           vertical_flip=True,
                                           #rotation_range=180,
                                           zoom_range=(1., 1.5),
                                           brightness_range=(0.8, 1.2),
                                           channel_shift_range=5),
                 flow_from_dir=dict(target_size=train_params['img_size'][:2],  # Only want width/height here
                                    color_mode='rgb',
                                    classes=['average', 'poor', 'poorest', 'rich', 'richest'],  # Keep this ordering, it should match class_weights
                                    batch_size=32,  # Want as large as GPU can handle, using batch-norm layers
                                    seed=42,  # Seed for random number generator
                                    save_to_dir=None))  # Set to visualize augmentations
