import os
from setuptools import setup


def get_long_description():
    this_directory = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    return long_description


def parse_requirements(file):
    return sorted(set(
        line.partition('#')[0].strip()
        for line in open(os.path.join(os.path.dirname(__file__), file))
    ) - set(''))


setup(
    name='eo-learn',
    python_requires='>=3.5',
    version='0.5.2',
    description='Earth observation processing framework for machine learning in Python',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/sentinel-hub/eo-learn',
    author='Sinergise EO research team',
    author_email='eoresearch@sinergise.com',
    license='MIT',
    packages=[],
    include_package_data=True,
    install_requires=[
        'eo-learn-core>=0.5.2',
        'eo-learn-coregistration>=0.5.0',
        'eo-learn-features>=0.5.0',
        'eo-learn-geometry>=0.5.0',
        'eo-learn-io>=0.5.0',
        'eo-learn-mask>=0.5.0',
        'eo-learn-ml-tools>=0.5.0',
        'eo-learn-visualization>=0.5.2'
    ],
    extras_require={
        'DEV': parse_requirements('requirements-dev.txt')
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ]
)
