import os

from setuptools import setup


def get_long_description():
    this_directory = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    # Removes lines with an image from description
    return "\n".join(line for line in long_description.split("\n") if not line.strip().startswith("<img"))


def parse_requirements(file):
    return sorted(
        set(line.partition("#")[0].strip() for line in open(os.path.join(os.path.dirname(__file__), file))) - set("")
    )


setup(
    name="eo-learn",
    python_requires=">=3.7",
    version="1.0.2",
    description="Earth observation processing framework for machine learning in Python",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/sentinel-hub/eo-learn",
    author="Sinergise EO research team",
    author_email="eoresearch@sinergise.com",
    license="MIT",
    packages=[],
    include_package_data=True,
    install_requires=[
        "eo-learn-core==1.0.2",
        "eo-learn-coregistration==1.0.2",
        "eo-learn-features==1.0.2",
        "eo-learn-geometry==1.0.2",
        "eo-learn-io==1.0.2",
        "eo-learn-mask==1.0.2",
        "eo-learn-ml-tools==1.0.2",
        "eo-learn-visualization==1.0.2",
    ],
    extras_require={"DEV": parse_requirements("requirements-dev.txt")},
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
        "Topic :: Scientific/Engineering",
    ],
)
