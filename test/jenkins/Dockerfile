FROM amazonlinux:latest
RUN yum -y install which unzip aws-cli python36 git
RUN yum -y python36 gcc64-c++ libzip-devel zlib-static createrepo aws-cli unzip
RUN pip-3.6 install  numpy pillow keras graphviz tensorflow==1.5 filelock boto3

