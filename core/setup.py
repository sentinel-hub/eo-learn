import os
from setuptools import setup, find_packages


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


def get_version():
    for line in open(os.path.join(os.path.dirname(__file__), 'eolearn', 'core', '__init__.py')):
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"').strip("'")
    return version


setup(
    name='eo-learn-core',
    python_requires='>=3.5',
    version=get_version(),
    description='Core Machine Learning Framework at Sinergise',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/sentinel-hub/eo-learn',
    author='Sinergise EO research team',
    author_email='eoresearch@sinergise.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    zip_safe=False
)
