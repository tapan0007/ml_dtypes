FROM ubuntu:16.04

RUN apt update -y && \
    apt install -y \
        createrepo \
        gcc-4.8 \
        g++-4.8 \
        git \
        graphviz \
        unzip \
        libboost-all-dev \
        cmake \
        glib* \
        libpixman-1-dev \
        build-essential \
        python-setuptools \
        python3-setuptools \
        python3-pip \
        libz-dev \
        pkg-config \
        libglib2.0-dev zlib1g-dev \
        swig \
        && \
    apt clean -y

RUN pip3 install --no-cache-dir \
        awscli \
        boto3 \
        filelock \
        graphviz \
        keras \
        numpy==1.15.4 \
        pillow \
        psutil \
        virtualenv \
        scikit-image \
        tensorflow==1.8 \
        pexpect \
        junit_xml \
        pytest \
	grpcio \
	protobuf \
	tabulate \
	retrying \
	allpairspy \
	hexdump \
        progress \
        lark \
        lark-parser \
        cmd2 \
        mako

COPY ssh-config /tmp/ssh-config
RUN cat /tmp/ssh-config >> /etc/ssh/ssh_config

ADD https://storage.googleapis.com/git-repo-downloads/repo /usr/bin/repo
RUN chmod a+rx /usr/bin/repo
