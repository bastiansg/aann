FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git-core \
    unzip \
    zsh \
    pkg-config \
    libcairomm-1.0-dev \
    python3-setuptools \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils

# Set the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

WORKDIR /tmp

# Install pip
ARG PIP_VERSION
RUN wget https://github.com/pypa/pip/archive/refs/tags/${PIP_VERSION}.zip \
    && unzip ${PIP_VERSION}.zip \
    && cd pip-${PIP_VERSION} \
    && python setup.py install

# Install graph-tool
RUN wget -qO - https://keys.openpgp.org/vks/v1/by-fingerprint/793CEFE14DBC851A2BFB1222612DEFB798507F25 | gpg --dearmor | tee /usr/share/keyrings/graph-tool.gpg > /dev/null \
    && echo "deb [ arch=amd64 signed-by=/usr/share/keyrings/graph-tool.gpg ] https://downloads.skewed.de/apt jammy main" | tee /etc/apt/sources.list.d/graph-tool.list \
    && apt update && apt-get install -y --no-install-recommends python3-graph-tool \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/tmp/* /var/lib/apt/lists/*

# Install pytorch & torchvision
ARG CUDA_VERSION
ARG PYTORCH_VERSION
RUN --mount=type=cache,target=/root/.cache \
    pip install torch==${PYTORCH_VERSION}+${CUDA_VERSION} \
    -f https://download.pytorch.org/whl/${CUDA_VERSION}/torch_stable.html

# Install aann
COPY ./src/aann /tmp/aann
RUN --mount=type=cache,target=/root/.cache \
    pip install ./aann

# Clean
RUN rm -rf /tmp/*

WORKDIR /root
