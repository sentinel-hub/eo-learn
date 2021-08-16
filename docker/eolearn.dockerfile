FROM python:3.8-buster

LABEL maintainer="Sinergise EO research team <eoresearch@sinergise.com>"
LABEL description="An official eo-learn docker image with a full eo-learn installation and Jupyter notebook."

RUN apt-get update && apt-get install -y \
        gcc \
        libgdal-dev \
        graphviz \
        proj-bin \
        libproj-dev \
        libspatialindex-dev \
    && apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip3 install --no-cache-dir pip --upgrade
RUN pip3 install --no-cache-dir shapely --no-binary :all:
RUN FORCE_CYTHON=1 pip install cartopy

WORKDIR /tmp

COPY core core
COPY coregistration coregistration
COPY features features
COPY geometry geometry
COPY io io
COPY mask mask
COPY ml_tools ml_tools
COPY visualization visualization
COPY setup.py README.md requirements-dev.txt ./

RUN pip3 install --no-cache-dir \
    ./core \
    ./coregistration \
    ./features \
    ./geometry \
    ./io[GEODB,METEOBLUE] \
    ./mask \
    ./ml_tools \
    ./visualization[FULL]

RUN pip3 install --no-cache-dir \
    . \
    rtree \
    jupyter

RUN rm -r ./*

ENV TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

WORKDIR /home/eolearner

EXPOSE 8888
CMD ["/usr/local/bin/jupyter", "notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0", \
     "--NotebookApp.token=''", "--allow-root"]
