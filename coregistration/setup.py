import os

from setuptools import find_packages, setup


def get_long_description():
    this_directory = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    return long_description


def parse_requirements(file):
    with open(os.path.join(os.path.dirname(__file__), file)) as requirements_file:
        return sorted({line.partition("#")[0].strip() for line in requirements_file} - set(""))


def get_version():
    path = os.path.join(os.path.dirname(__file__), "eolearn", "coregistration", "__init__.py")
    with open(path) as version_file:
        for line in version_file:
            if line.find("__version__") >= 0:
                version = line.split("=")[1].strip()
                version = version.strip('"').strip("'")
    return version


setup(
    name="eo-learn-coregistration",
    python_requires=">=3.7",
    version=get_version(),
    description="A collection of image co-registration EOTasks",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/sentinel-hub/eo-learn",
    project_urls={
        "Documentation": "https://eo-learn.readthedocs.io",
        "Source Code": "https://github.com/sentinel-hub/eo-learn",
        "Bug Tracker": "https://github.com/sentinel-hub/eo-learn/issues",
        "Forum": "https://forum.sentinel-hub.com",
    },
    author="Sinergise EO research team",
    author_email="eoresearch@sinergise.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
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
