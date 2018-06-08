FROM amazonlinux:latest

RUN yum update -y && \
    yum install -y \
        aws-cli \
        createrepo \
        gcc48-c++ \
        gcc64-c++ \
        git \
        libzip-devel \
        python36 \
        python36-devel \
        unzip \
        which \
        boost-devel \
        cmake \
        zlib-devel \
        glib* \
        pixman \
        pixman-devel \
        zlib-static && \
    yum groupinstall -y "Development tools" && \
    yum clean all -y

RUN pip-3.6 install --no-cache-dir \
        boto3 \
        filelock \
        graphviz \
        keras \
        numpy \
        pillow \
        psutil \
        virtualenv \
        scikit-image \
        tensorflow==1.5 \
        pexpect

COPY ssh-config /tmp/ssh-config
RUN cat /tmp/ssh-config >> /etc/ssh/ssh_config

ADD https://storage.googleapis.com/git-repo-downloads/repo /usr/bin/repo
RUN chmod a+x /usr/bin/repo
