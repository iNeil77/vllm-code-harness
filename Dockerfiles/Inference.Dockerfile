FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV CUDA_HOME="/usr/local/cuda" \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    PYTHONUNBUFFERED=1 \
    GOPATH="/container/go" \
    GO111MODULE="off" \
    CARGO_HOME="/container/cargo" \
    RUSTUP_HOME="/container/rustup"

# Setup System Utilities and Languages: C, C++, Haskell, Java, Lua, OCaml, Perl, R, Ruby, Scala and lang-specific dependencies like Boost (C++)
RUN apt update --yes --quiet \
    && apt upgrade --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
        apache2 \
        apache2-bin \
        apache2-data \
        apache2-utils \
        apt-transport-https \
        apt-utils \
        autoconf \
        automake \
        bc \
        bison \
        build-essential \
        ca-certificates \
        check \
        cmake \
        curl \
        dmidecode \
        emacs \
        g++ \
        gcc \
        ghc \
        git \
        gnupg \
        htop \
        iproute2 \
        iotop \
        jq \
        kmod \
        libaio-dev \
        libapr1-dev \
        libboost-all-dev \
        libcurl4-openssl-dev \
        libffi-dev \
        libgdbm-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libibverbs-dev \
        libncurses5-dev \
        libnuma-dev \
        libnuma1 \
        libomp-dev \
        libreadline-dev \
        libsm6 \
        libssl-dev \
        libsubunit-dev \
        libsubunit0 \
        libtest-deep-perl \
        libtool \
        libxext6 \
        libxrender-dev \
        libyaml-dev \
        lsb-release \
        lsof \
        lua-unit \
        lua5.3 \
        make \
        moreutils \
        net-tools \
        ninja-build \
        ocaml \
        ocaml-interp \
        openjdk-21-jdk-headless \
        openjdk-21-jre-headless \
        openssh-client \
        openssh-server \
        openssl \
        pkg-config \
        python3-dev \
        r-base \
        racket \
        rlwrap \
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

# Setup Perl testing dependencies
RUN perl -MCPAN -e 'install Test::Deep' \
    && perl -MCPAN -e 'install Test::Differences' \
    && perl -MCPAN -e 'install Data::Compare'

# Setup R testing dependencies
RUN R -e "install.packages('testthat', repos='http://cran.rstudio.com/')" \
    && R -e "install.packages('devtools', repos='http://cran.rstudio.com/')"

# Setup Php and its testing dependencies
RUN add-apt-repository ppa:ondrej/php \
    && apt update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends php8.4 \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
        php8.4-bcmath \
        php8.4-cgi \
        php8.4-cli \
        php8.4-common \
        php8.4-curl \
        php8.4-fpm \
        php8.4-gd \
        php8.4-gettext \
        php8.4-intl \
        php8.4-mbstring \
        php8.4-mysql \
        php8.4-mysqlnd \
        php8.4-opcache \
        php8.4-pdo \
        php8.4-pgsql \
        php8.4-readline \
        php8.4-sqlite3 \
        php8.4-xml \
        php8.4-zip

# Clojure
RUN curl -L -O https://github.com/clojure/brew-install/releases/latest/download/linux-install.sh
RUN chmod +x linux-install.sh
RUN ./linux-install.sh --prefix /clojure
ENV PATH="/clojure/bin:${PATH}"
RUN clojure -P

# Dart
RUN wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg  --dearmor -o /usr/share/keyrings/dart.gpg
RUN echo 'deb [signed-by=/usr/share/keyrings/dart.gpg arch=amd64] https://storage.googleapis.com/download.dartlang.org/linux/debian stable main' | tee /etc/apt/sources.list.d/dart_stable.list
RUN apt-get update -yqq && apt-get install -yqq dart

# Setup Go and its testing dependencies
RUN add-apt-repository --yes ppa:longsleep/golang-backports \
    && apt update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends golang-1.18 \
    && ln -s /usr/lib/go-1.18/bin/go /usr/bin/go \
    && go get github.com/stretchr/testify/assert

# Setup JS/TS and auxiliary tools
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash - \
    && DEBIAN_FRONTEND=noninteractive apt install -y nodejs \
    && npm install -g lodash \
    && npm install -g typescript

# Setup Dlang
RUN wget https://netcologne.dl.sourceforge.net/project/d-apt/files/d-apt.list -O /etc/apt/sources.list.d/d-apt.list \
    && apt update --allow-insecure-repositories \
    && apt -y --allow-unauthenticated install --reinstall d-apt-keyring \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -yqq dmd-compiler dub

# Setup C# and dotnet runtime
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
    && echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list \
    && apt update -yqq \
    && DEBIAN_FRONTEND=noninteractive apt install -yqq mono-devel dotnet-sdk-8.0 dotnet-runtime-8.0

# Setup Swift
RUN curl https://download.swift.org/swift-5.10.1-release/ubuntu2204/swift-5.10.1-RELEASE/swift-5.10.1-RELEASE-ubuntu22.04.tar.gz | tar xz -C /container/
ENV PATH="/container/swift-5.10.1-RELEASE-ubuntu22.04/usr/bin:${PATH}"

# Setup Julia
RUN curl https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz | tar xz -C /container/
ENV PATH="/container/julia-1.10.4/bin:${PATH}"

# Install Java testing dependencies
RUN mkdir /container/multipl-e \
    && wget https://repo.mavenlibs.com/maven/org/javatuples/javatuples/1.2/javatuples-1.2.jar -O /container/multipl-e/javatuples-1.2.jar

# Setup base Python to bootstrap Mamba
RUN add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-lib2to3 \
        python3.11-gdbm \
        python3.11-tk \
        pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 999 \
    && update-alternatives --config python3 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools

# Setup Mamba environment and Rust
RUN wget -O /tmp/Miniforge.sh https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-24.3.0-0-Linux-x86_64.sh \
    && bash /tmp/Miniforge.sh -b -p /Miniforge \
    && source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba update -y -q -n base -c defaults mamba \
    && mamba create -y -q -n inference python=3.12 setuptools=69.5.1 cxx-compiler=1.7.0 \
    && mamba activate inference \
    && mamba install -y -q -c conda-forge \
        charset-normalizer \
        gputil \
        ipython \
        mkl \
        mkl-include \
        'numpy<2.0.0' \
        pandas \
        rust=1.80.1 \
        scikit-learn \
        wandb \
    && mamba install -y -q -c pytorch -c nvidia magma-cuda124 pytorch==2.4.0 pytorch-cuda=12.4 \
    && mamba clean -a -f -y

# Install vllm and eval-harness dependencies
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate inference \
    && pip install 'accelerate>=0.13.2' \
        camel_converter \
        cdifflib \
        'datasets>=2.6.1,<3.0.0' \
        diff_match_patch \
        'evaluate>=0.3.0' \
        'fsspec<2023.10.0' \
        'huggingface_hub>=0.11.1' \
        hf_transfer \
        jsonlines \
        maturin \
        'mosestokenizer==1.0.0' \
        ninja \
        nltk \
        openai \
        packaging \
        patchelf \
        peft \
        protobuf \
        py7zr \
        requests \
        'rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1' \
        'sentencepiece!=0.1.92' \
        seqeval \
        'setuptools>=49.4.0' \
        termcolor \
        'transformers==4.44.1' \
        'vllm==0.6.1.post2' \
        wheel

# Install Flash Attention
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate inference \
    && export MAX_JOBS=$(($(nproc) - 2)) \
    && pip install --no-cache-dir ninja packaging \
    && pip install flash-attn==2.6.3 --no-build-isolation
