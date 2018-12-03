## Instructions for running examples on Amazon Sagemaker

[Amazon Sagemaker](https://aws.amazon.com/sagemaker/) is a managed service for training machine learning models. Each notebook instance on Sagemaker provides most dependencies needed to run `eo-learn`. Here's how to run our example Jupyter Notebooks on Sagemaker:

- Start an [Amazon SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- Upload any of our example Jupyter Notebooks.
- Add a new first cell to install extra dependencies: `!pip install eo-learn-io geopandas tqdm`
- Thats it! Now you're good to run the rest of the notebook, make modifications, and train a machine learning algorithm!
