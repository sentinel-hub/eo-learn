# Land Use and Land Cover (LULC) classification with [fast.ai](https://docs.fast.ai/)

This example showcases how it’s possible to perform an automatic LULC classification with deep learning, using the cutting-edge fast.ai deep learning package, to work with `eo-learn`. The classification is performed on Sentinel-2 scenes from 2017. The underline deep learning is Dynamic-U-net, we used ResNet50 as the encoder in this case.

The workflow breaks down into training dataset generation, model training, and prediction. The notebook contains a detailed walkthrough. If this your first time working with LULC case using SentinelHub, eo-learn, or deep learning, you may find the following things helpful.

### Setup

In order to run the example, the `fastai` Python library need to be installed as follows:

```bash
pip install fastai
```

All the other dependecies are already installed with `eo-learn`.

### Step 1. Training data generation

We used eo-learn package to fetch Sentinel-2 imagery from SentinelHub, and generate training image tiles as PNG to serve the deep learning model. Here are a few README if you want to learn more about how the training dataset was generated eo-learn and SentinelHub python packages: 1) [image file generation](https://github.com/sentinel-hub/eo-learn/tree/master/examples/land-cover-map) from a given shapefile/geojson of your area of interest; 2) [using EOPatches](https://github.com/sentinel-hub/eo-learn/tree/master/examples/land-cover-map) to save the Sentinel-2 imagery bands that is valid for the model.

We create a util script to read bands information and burned them into PNGs to feed the model on the cloud.

### Step 2. Training model on the cloud

If it’s your first try to train a deep learning model on the cloud, we recommend you using a GPU machine to train the model. You can either use AWS EC2 machine (Deep learning AMI ami-6d720012), SageMaker ( detail setup can see [this instruction](https://github.com/sentinel-hub/eo-learn/blob/master/examples/tree-cover-keras/sagemaker.md)), Google colab, or other cloud providers’ services. We use AWS Deep Learn AMI p2.xlarge machine to run the notebook.

The total training and prediction will take from 5 to 10 hours, depends on how long you want to train the model, will cost you from $5 to $10. If you used our pre-trained model weight “stage-8-50-ind“ (It’s in [S3 bucket](https://s3.amazonaws.com/query-planet-fastai-model)), it will take less than 5 hours.  Google Colab provides credit for first users and it’d be free.


### Step 3. Prediction and model inference

Prediction/model inference is pretty straight forward: we are feeding image tiles/PNGs to the trained model, and predict the LULC, and we visualized it with Matplotlib. You can write a utility function to write the prediction back to associated EOPatch, see [this tutorial](https://eo-learn.readthedocs.io/en/latest/examples/land-cover-map/SI_LULC_pipeline.html#8.-Visualization-of-the-results).
