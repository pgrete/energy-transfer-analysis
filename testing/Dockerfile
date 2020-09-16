FROM ubuntu:20.04

RUN apt-get clean && apt-get update -y && \
    DEBIAN_FRONTEND="noninteractive" TZ=America/New_York apt-get install -y --no-install-recommends python3-scipy libhdf5-openmpi-103 openmpi-bin python3-pip python3-h5py wget libfftw3-bin libfftw3-double3 libfftw3-long3 libfftw3-quad3 libfftw3-single3 libfftw3-dev build-essential git python3-dev python3-matplotlib python3-sympy python3-ipython ipython3 openssh-client libpython3-stdlib

RUN apt-get clean

RUN pip3 install yt

RUN pip3 install mpi4py-fft

ENV OMPI_ALLOW_RUN_AS_ROOT=1

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN mkdir -p /data/enzo && \
    cd /data && \
    wget https://pgrete.de/dl/EnTrans/EnTransTesting-DD0024.tgz && \
    tar xzf EnTransTesting-DD0024.tgz -C enzo && \
    rm EnTransTesting-DD0024.tgz

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
