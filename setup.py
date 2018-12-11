import os
from setuptools import setup


def get_long_description():
    this_directory = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    return long_description


setup(name='eo-learn',
      python_requires='>=3.5',
      version='0.4.0',
      description='Earth observation processing framework for machine learning in Python',
      long_description=get_long_description(),
      long_description_content_type='text/markdown',
      url='https://github.com/sentinel-hub/eo-learn',
      author='Sinergise EO research team',
      author_email='info@sinergise.com',
      license='MIT',
      packages=[],
      include_package_data=True,
      install_requires=[
          'eo-learn-core>=0.4.0',
          'eo-learn-coregistration>=0.4.0',
          'eo-learn-features>=0.4.0',
          'eo-learn-geometry>=0.4.0',
          'eo-learn-io>=0.4.0',
          'eo-learn-mask>=0.4.0',
          'eo-learn-ml-tools>=0.4.0'
      ],
      zip_safe=False)
