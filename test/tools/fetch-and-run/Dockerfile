FROM amazonlinux:latest
RUN yum -y install which unzip aws-cli python36
RUN pip-3.6 install  boto3
ADD alinux_deps_install /tmp/alinux_deps_install
RUN /tmp/alinux_deps_install

ADD fetch_and_run.sh /usr/local/bin/fetch_and_run.sh
WORKDIR /tmp
ENTRYPOINT ["/usr/local/bin/fetch_and_run.sh"]
