
"""
Note: After we have the tf-serving image running, we don't actually need following script.
pred_config.py

List some configuration parameters for prediction
"""
import os
from os import path as op

preds_dir = op.join(os.getcwd(), "preds")
plot_dir = op.join(os.getcwd(), "plots")
ckpt_dir = op.join(os.getcwd(), "models")


# Params to run inference on some tiles in S3
pred_params = dict(model_time='0813_192409', #201910_L5.39
                   single_batch_size=16,  # Number of images seen by a single GPU
                   n_gpus=1,
                   deci_prec=2)  # Number of decimal places in prediction precision
pred_params.update(dict(model_arch_fname='{}_arch.yaml'.format(pred_params['model_time']),
                        model_params_fname='{}_params.yaml'.format(pred_params['model_time']),
                        model_weights_fname='{}_L1.20_E13_weights.h5'.format(pred_params['model_time'])))
