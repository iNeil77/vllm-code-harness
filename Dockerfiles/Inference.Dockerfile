FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV CUDA_HOME="/usr/local/cuda" \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    PYTHONUNBUFFERED=1 \
    GOPATH="/container/go" \
    GO111MODULE="off" \
    CARGO_HOME="/container/cargo" \
    RUSTUP_HOME="/container/rustup"

# Setup System Utilities and Languages: C, C++, Java, Lua, Perl, R, Ruby, Scala and lang-specific dependencies like Boost (C++)
RUN apt update --yes --quiet \
    && apt upgrade --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
        apt-utils \
        autoconf \
        automake \
        bc \
        build-essential \
        ca-certificates \
        check \
        cmake \
        curl \
        dmidecode \
        emacs \
        g++ \
        gcc \
        git \
        htop \
        iproute2 \
        jq \
        kmod \
        libaio-dev \
        libboost-all-dev \
        libcurl4-openssl-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libibverbs-dev \
        libnuma-dev \
        libnuma1 \
        libomp-dev \
        libsm6 \
        libssl-dev \
        libsubunit-dev \
        libsubunit0 \
        libtest-deep-perl \
        libtool \
        libxext6 \
        libxrender-dev \
        lsof \
        lua-unit \
        lua5.3 \
        make \
        moreutils \
        net-tools \
        ninja-build \
        openjdk-21-jdk-headless \
        openjdk-21-jre-headless \
        openssh-client \
        openssh-server \
        openssl \
        php-cli \
        pkg-config \
        python3-dev \
        r-base \
        ruby \
        scala \
        software-properties-common \
        sudo \
        tmux \
        unzip \
        util-linux \
        vim \
        wget \
        zlib1g-dev \
    && apt autoremove \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Setup Go and its testing dependencies
RUN add-apt-repository --yes ppa:longsleep/golang-backports \
    && apt update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends golang-1.18 \
    && ln -s /usr/lib/go-1.18/bin/go /usr/bin/go \
    && go get github.com/stretchr/testify/assert

# Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH=/container/cargo/bin:$PATH

# JS/TS
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - \
    && apt install -y nodejs \
    && npm install -g typescript

# Dlang
RUN wget https://netcologne.dl.sourceforge.net/project/d-apt/files/d-apt.list -O /etc/apt/sources.list.d/d-apt.list \
    && apt update --allow-insecure-repositories \
    && apt -y --allow-unauthenticated install --reinstall d-apt-keyring \
    && apt update \
    && apt install -yqq dmd-compiler dub

# C#
RUN apt install gnupg ca-certificates \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
    && echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list \
    && apt update -yqq \
    && apt install -yqq mono-devel

# Swift
RUN curl https://download.swift.org/swift-5.7-release/ubuntu2204/swift-5.7-RELEASE/swift-5.7-RELEASE-ubuntu22.04.tar.gz | tar xz
ENV PATH="/swift-5.7-RELEASE-ubuntu22.04/usr/bin:${PATH}"

# Julia
RUN curl https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz | tar xz
ENV PATH="/julia-1.8.2/bin:${PATH}"

# JavaTuples
RUN mkdir /container/multipl-e \
    && wget https://repo.mavenlibs.com/maven/org/javatuples/javatuples/1.2/javatuples-1.2.jar -O /container/multipl-e/javatuples-1.2.jar

# Setup base Python to bootstrap Mamba
RUN add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3.10-lib2to3 \
        python3.10-gdbm \
        python3.10-tk \
        pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && pip install --upgrade pip

# Setup Mamba environment
RUN wget -O /tmp/Miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/Miniforge.sh -b -p /Miniforge \
    && source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba update -y -q -n base -c defaults mamba \
    && mamba create -y -q -n inference python=3.11 setuptools=69.5.1 \
    && mamba activate inference \
    && mamba install -y -q -c conda-forge \
        charset-normalizer \
        gputil \
        ipython \
        mkl \
        mkl-include \
        numpy \
        pandas \
        scikit-learn \
        wandb \
    && mamba install -y -q -c pytorch magma-cuda121 \
    && mamba clean -a -f -y

# Install vllm and eval-harness dependencies
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate inference \
    && pip install 'accelerate>=0.13.2' \
        camel_converter \
        cdifflib \
        'datasets>=2.6.1' \
        diff_match_patch \
        'evaluate>=0.3.0' \
        'fsspec<2023.10.0' \
        'huggingface_hub>=0.11.1' \
        jsonlines \
        'mosestokenizer==1.0.0' \
        ninja \
        nltk \
        openai \
        packaging \
        peft \
        protobuf \
        py7zr \
        requests \
        'rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1' \
        'sentencepiece!=0.1.92' \
        seqeval \
        'setuptools>=49.4.0' \
        termcolor \
        'transformers==4.41.2' \
        'vllm==0.5.0' \
        wheel

# Install Flash Attention
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate inference \
    && export MAX_JOBS=$(($(nproc) - 2)) \
    && pip install --no-cache-dir ninja packaging \
    && pip install flash-attn==2.5.8 --no-build-isolation