# Use nvidia/cuda image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y \
    gcc git wget \
    ffmpeg libsm6 libxext6 default-jdk \
    python3.8 python3.8-dev python3-pip

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py

# install requirements for API
COPY cli.py cli.py
COPY deepliif deepliif
COPY setup.py setup.py

RUN pip install .
