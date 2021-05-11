# Use nvidia/cuda image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# set bash as current shell
RUN chsh -s /bin/bash

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 libgl1-mesa-dev git mercurial subversion && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH


# setup conda virtual environment
COPY ./environment.yml /tmp/environment.yml
RUN conda update conda \
    && conda env create --name pytorch_env -f /tmp/environment.yml
    
RUN echo "conda activate pytorch_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/pytorch_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $pytorch_env

WORKDIR XXX/DeepLIIF
COPY . .
