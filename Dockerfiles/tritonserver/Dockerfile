FROM nvcr.io/nvidia/l4t-ml:r32.4.3-py3
SHELL ["/bin/bash", "-c"]

# https://github.com/NVIDIA/triton-inference-server/releases/download/v2.0.0/tritonserver2.0.0-jetpack4.4ga.tgz
RUN mkdir /root/tritonserver
COPY tritonserver2.0.0-jetpack4.4ga.tgz /root/tritonserver
WORKDIR /root/tritonserver
RUN tar -zxf tritonserver2.0.0-jetpack4.4ga.tgz
RUN rm tritonserver2.0.0-jetpack4.4ga.tgz

RUN apt update
RUN apt install -y --no-install-recommends software-properties-common autoconf automake build-essential cmake git libb64-dev libgoogle-glog0v5 libre2-dev libssl-dev libtool libboost-dev libcurl4-openssl-dev rapidjson-dev patchelf zlib1g-dev libffi-dev
RUN apt install -y python3-pip
RUN mkdir ~/.pip
RUN echo $'[global] \n\
trusted-host = mirrors.aliyun.com \n\
index-url = https://mirrors.aliyun.com/pypi/simple' > ~/.pip/pip.conf
RUN pip3 install --upgrade cython
# RUN pip3 install --upgrade clients/python/triton*.whl

CMD ./bin/tritonserver --model-repository=/model
# docker run --rm --runtime nvidia -p8000:8000 -p8001:8001 -p8002:8002 -v /home/wcg/edge/model_repository:/model nx-tritonserver
