# Superresolution using fastai

This example shows how to create a machine learning model to "predict" higher resolution images from medium resolution Sentinel-2 imagery. The example is made up of two notebooks:
- [DataPrep](DataPrep.ipynb): uses `eo-learn` to create corresponding image chips of Sentinel-2 imagery and Digital Globe imagery via [Spacenet](https://spacenetchallenge.github.io/AOI_Lists/AOI_3_Paris.html).
- [Train](Train.ipynb): uses `fastai` to train a machine learning model from these images. It primarily follows the example from [Lesson 7](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres.ipynb) of the fast.ai course
