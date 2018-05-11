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
        zlib-static && \
    yum clean all -y

RUN pip-3.6 install --no-cache-dir \
        boto3 \
        filelock \
        graphviz \
        keras \
        numpy \
        pillow \
        psutil \
        scikit-image \
        tensorflow==1.5

COPY ssh-config /tmp/ssh-config
RUN cat /tmp/ssh-config >> /etc/ssh/ssh_config

ADD https://storage.googleapis.com/git-repo-downloads/repo /usr/bin/repo
RUN chmod a+x /usr/bin/repo
