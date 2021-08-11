FROM sentinelhub/eolearn:latest

RUN apt-get update && apt-get install -y \
        ffmpeg \
    && apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        ffmpeg-python \
        ipyleaflet

COPY ./examples ./examples
COPY ./example_data ./example_data
