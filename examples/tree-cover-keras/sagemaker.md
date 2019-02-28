## Instructions for running examples on Amazon SageMaker

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a managed service for training machine learning models. Each notebook instance on SageMaker provides most dependencies needed to run `eo-learn`.

There are roughly three ways to our example Jupyter Notebooks on SageMaker:

### Install the Dependencies Manually, Notebook Training

- Start an [Amazon SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- Upload any of our example Jupyter Notebooks.
- Add a new first cell to install extra dependencies: `!pip install eo-learn-io geopandas tqdm`
- Thats it! Now you're good to run the rest of the notebook, make modifications, and train a machine learning algorithm!

### Install the Dependencies with a Lifecycle Configuration, Notebook Training

- Before starting a Notebook Instance, add a Lifecycle Configuration. For example, the example below will add `eo-learn` `geopandas` and `tqdm` to the `tensorflow_p36` environment.

```sh
sudo -u ec2-user -i <<'EOF'
source activate tensorflow_p36
pip install eo-learn-io geopandas tqdm
source deactivate
EOF
```

 - Configure this script to run on instance creation:

<img width="1350" alt="amazon_sagemaker" src="https://user-images.githubusercontent.com/7108211/51563298-f9993200-1e59-11e9-9c03-fe1c2e457c8c.png">

- Run the notebook as in the above example

### Submit a Training Script to SageMaker

Sagemaker also provides the ability to train a model on a separate instance and deploy on sagemaker. Here are the main steps:
1. **Save data to S3**: Instead of using all the data in a single notebook instance, we can use `eo-learn` to download and process the data and write it to S3:

```python
import sagemaker
from eolearn.core import LinearWorkflow, SaveToDisk

sagemaker_session = sagemaker.Session()

...

# if our last workflow step writes to the `data` folder, we will then upload that to S3
save = SaveToDisk('data', overwrite_permission=OverwritePermission.OVERWRITE_PATCH, compress_level=2)
workflow = LinearWorkflow(..., save)

for task in tasks:
    workflow.execute(task)

inputs = sagemaker_session.upload_data(path='data/', key_prefix='example/eo-learn')
```
2. **Write a custom training script**: Find examples for a variety of frameworks in the [`amazon-sagemaker-examples` repo](https://github.com/awslabs/amazon-sagemaker-examples). Save this script as `custom_script.py` within the notebook. The custom portion needed for `eo-learn` is reading data from `.npy.gz` files:

```python
import gzip
import numpy as np
from glob import glob

...

files = glob('train_dir/*')

x_train = np.empty((len(files), 256, 256, 3))
for i, file in enumerate(files):
  file = gzip.GzipFile('TRUE_COLOR_S2A.npy.gz', 'r')
  x_train[i] = np.load(file)
```

3. **Invoke the training script**: Now we can invoke the training script on a separate, and potentially more powerful, instance from the notebook:

```python
from sagemaker import get_execution_role
role = get_execution_role()
from sagemaker.tensorflow import TensorFlow

custom_estimator = TensorFlow(entry_point='custom_script.py',
                               role=role,
                               framework_version='1.12.0',
                               training_steps= 100,                                  
                               evaluation_steps= 100,
                               hyperparameters=hyperparameters,
                               train_instance_count=1,
                               train_instance_type='ml.p3.2xlarge')

custom_estimator.fit(inputs)
```

4. **Deploy the trained model**: As a bonus, this makes it very easy to deploy the trained model which can serve real-time prediction requests:

```python
custom_predictor = custom_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
custom_predictor.predict(test_image)
```

Check out the [full example](tree-cover-keras-sagemaker.ipynb) for more help.
