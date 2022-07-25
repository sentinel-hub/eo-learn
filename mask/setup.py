import os

from setuptools import find_packages, setup


def get_long_description():
    this_directory = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    return long_description


def parse_requirements(file):
    return sorted(
        set(line.partition("#")[0].strip() for line in open(os.path.join(os.path.dirname(__file__), file))) - set("")
    )


def get_version():
    for line in open(os.path.join(os.path.dirname(__file__), "eolearn", "mask", "__init__.py")):
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"').strip("'")
    return version


setup(
    name="eo-learn-mask",
    python_requires=">=3.7",
    version=get_version(),
    description="A collection of masking EOTasks and utilities",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/sentinel-hub/eo-learn",
    author="Sinergise EO research team",
    author_email="eoresearch@sinergise.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "eolearn.mask": [
            "models/pixel_s2_cloud_detector_lightGBM_v0.2.txt",
            "models/ssim_s2_cloud_detector_lightGBM_v0.2.txt",
        ]
    },
    install_requires=parse_requirements("requirements.txt"),
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
