FROM sentinelhub/eolearn:latest

LABEL description=\
"An official eo-learn docker image with a full eo-learn installation, Jupyter notebook, all \
example notebooks, and some additional dependencies required to run those notebooks."

RUN apt-get update && apt-get install -y \
        ffmpeg \
    && apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        ffmpeg-python \
        ipyleaflet

COPY ./examples ./examples
COPY ./example_data ./example_data
