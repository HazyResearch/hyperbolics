# syntax = docker/dockerfile:experimental
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
	 emacs \
         parallel \
         ca-certificates \
         libjpeg-dev \
	 hdf5-tools \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


RUN mkdir ~/.parallel && touch ~/.parallel/will-cite

RUN curl -o ~/miniconda.sh -L -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl && \
     /opt/conda/bin/conda install -c soumith magma-cuda90 && \
     /opt/conda/bin/conda clean -ya 
ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available
WORKDIR /opt/pytorch
COPY . .
#RUN conda install pytorch torchvision -c pytorch

#
# Now install the julia dependencies.
#
ADD requirements.txt /app/
WORKDIR /app
#RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip cache list
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip install -r requirements.txt

ADD . /app 

WORKDIR /opt/julia
RUN conda install networkx joblib

RUN apt-get update && apt-get install -y curl
RUN mkdir /julia
RUN curl -L https://julialang-s3.julialang.org/bin/linux/x64/0.7/julia-0.7.0-linux-x86_64.tar.gz | tar -C /julia --strip-components=1  -xzf -
ENV PATH "/julia/bin:$PATH"
#RUN julia -e "using Pkg; Pkg.init()"
COPY setup.jl /julia/setup.jl
RUN julia /julia/setup.jl

WORKDIR /root/hyperbolics
ENV PYTHONPATH /root/hyperbolics
