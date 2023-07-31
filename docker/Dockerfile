# Use CUDA and cuDNN images from gitlab.com/nvidia/cuda as base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ARG CUDA_ARCH="6.0 6.1 7.0 7.5 8.0 8.6"

###############################################################################
# Install developing tools & dependencies
###############################################################################

# Change the default shell to Bash
SHELL ["/bin/bash", "--login", "-c"]

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -qq -y --no-install-recommends \
    build-essential gcc g++ cmake cmake-curses-gui make checkinstall binutils \
    ca-certificates apt-transport-https software-properties-common \
    freeglut3 freeglut3-dev libglfw3 libglfw3-dev libglew-dev libfreeimage-dev libglu1-mesa libglu1-mesa-dev libx11-dev libxmu-dev libxi-dev libgl1-mesa-dev \
    wget curl git vim tmux unzip zlib1g libzmq3-dev zsh sudo

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -qq -y --no-install-recommends \
    pkg-config qt5-default libgtk2.0-dev libjpeg8-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libv4l-dev libxvidcore-dev libx264-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libblas-dev liblapack-dev liblapacke-dev gfortran yasm libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev doxygen

# Install dependencies for PCL
RUN apt-get update && apt-get install -qq -y --no-install-recommends \
    libboost-all-dev libflann-dev libqhull-dev libgtest-dev libpcap-dev proj-data libproj-dev libxmu-dev libxi-dev libopenni-dev libopenni2-dev libopenni-sensor-primesense-dev

# Clean up
RUN apt-get -y autoremove && apt-get clean autoclean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Reset noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=dialog

###############################################################################
# Build & Install Packages
###############################################################################

# Change the default shell to Zsh
SHELL ["/bin/zsh", "--login", "-c"]

# Set environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}

# Set working directory
WORKDIR /

# Install Eigen3
ARG EIGEN_VERSION="3.4.0"
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz -O eigen3.tar.gz && tar -xzf eigen3.tar.gz && rm eigen3.tar.gz && mv eigen-${EIGEN_VERSION} eigen3 \
    && cd eigen3 && mkdir build && cd build \
    && cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++14 \
    && make install \
    && cd ../../ && rm -rf eigen3

# Install Yaml-CPP
ARG YAML_CPP_VERSION="0.7.0"
RUN wget -q https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-${YAML_CPP_VERSION}.tar.gz -O yaml-cpp.tar.gz && tar -xzf yaml-cpp.tar.gz && rm yaml-cpp.tar.gz && mv yaml-cpp-yaml-cpp-${YAML_CPP_VERSION} yaml-cpp \
    && cd yaml-cpp && mkdir build && cd build \
    && cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++14 -DBUILD_TESTING=OFF -DINSTALL_GTEST=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_BUILD_SHARED_LIBS=ON \
    && make -j$(nproc) \
    && make install \
    && cd ../../ && rm -rf yaml-cpp

# Install Pybind11
ARG PYBIND11_VERSION="2.10.0"
RUN wget -q https://github.com/pybind/pybind11/archive/refs/tags/v${PYBIND11_VERSION}.tar.gz -O pybind11.tar.gz && tar -xzf pybind11.tar.gz && rm pybind11.tar.gz && mv pybind11-${PYBIND11_VERSION} pybind11 \
    && cd pybind11 && mkdir build && cd build \
    && cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++14 -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF \
    && make -j$(nproc) \
    && make install \
    && cd ../../ && rm -rf pybind11

# Install OpenCV with CUDA support
ARG OPENCV_VERSION="3.4.15"
RUN wget -q https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz -O opencv.tar.gz && tar -xzf opencv.tar.gz && rm opencv.tar.gz && mv opencv-${OPENCV_VERSION} opencv \
    && wget -q https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz -O opencv_contrib.tar.gz && tar -xzf opencv_contrib.tar.gz && rm opencv_contrib.tar.gz && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib \
    && cd opencv && mkdir build && cd build \
    && cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules -DCMAKE_CXX_FLAGS=-std=c++14 -DOPENCV_ENABLE_NONFREE=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_MATLAB=OFF -DWITH_QT=ON -DWITH_OPENCL=OFF -WITH_OPENEXR=ON -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENMP=ON -DMKL_WITH_OPENMP=ON -DWITH_CUDA=ON -DCUDA_FAST_MATH=ON -DENABLE_FAST_MATH=ON -DBUILD_CUDA_STUBS=OFF -DBUILD_DOCS=OFF -DOPENCV_DNN_OPENCL=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_opencv_cudacodec=OFF -DBUILD_opencv_xfeatures2d=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    && make -j$(nproc) \
    && make install \
    && cd ../../ && rm -rf opencv opencv_contrib

# Set environment variables
ENV OPENCV_IO_ENABLE_OPENEXR 1

# Install PCL with CUDA support
ARG PCL_VERSION="1.10.1"
RUN wget -q https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-${PCL_VERSION}.tar.gz -O pcl.tar.gz && tar -xzf pcl.tar.gz && rm pcl.tar.gz && mv pcl-pcl-${PCL_VERSION} pcl \
    &&cd pcl && mkdir build && cd build \
    && cmake .. -Wno-dev -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=OFF -DBUILD_CUDA=OFF -DBUILD_apps=OFF -DBUILD_examples=OFF -DBUILD_global_tests=OFF -DBUILD_simulation=OFF -DCUDA_BUILD_EMULATION=OFF -DCMAKE_CXX_FLAGS=-std=c++14 -DPCL_ENABLE_SSE=ON -DPCL_SHARED_LIBS=ON \
    && make -j$(nproc) \
    && make install \
    && cd ../../ && rm -rf pcl

# Install miniconda3
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh -b -p /opt/conda && rm miniconda.sh
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda config --set ssl_verify no \
    && conda config --add channels conda-forge \
    && conda config --remove channels defaults \
    && conda update -n base -c conda-forge conda -qy \
    && conda install -n base -c conda-forge mamba -qy \
    && conda clean -afqy

# Add non-root user
ARG USERNAME="my_user"
ARG UID
ARG GID
SHELL ["/bin/bash", "--login", "-c"]
RUN addgroup --gid $GID ${USERNAME}
RUN adduser --disabled-password --home /home/${USERNAME} --shell /bin/zsh --gecos '' --uid $UID --gid $GID ${USERNAME} \ 
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    # Change folder ownership to non-root user
    && chown -R ${USERNAME}:${USERNAME} /opt/conda

########## Run container as non-root user from here onwards ###################
SHELL ["/bin/zsh", "--login", "-c"]
USER ${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR $HOME

# Install oh-my-zsh
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && sed -i 's/robbyrussell/candy/g' ${HOME}/.zshrc && zstyle ':omz:update' mode disabled

# Create new conda environment
ARG CONDA_ENV_NAME="py39"
ARG CONDA_ENV_PATHON_VERSION="3.9"
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ${HOME}/.zshrc && source ${HOME}/.zshrc \
    && conda create -n ${CONDA_ENV_NAME} python=${CONDA_ENV_PATHON_VERSION} -qy \
    && echo "conda activate ${CONDA_ENV_NAME}" >> ${HOME}/.zshrc

# Install packages from requirements.txt
COPY requirements.txt $HOME/requirements.txt
RUN source ${HOME}/.zshrc && python -m pip install --quiet --no-cache-dir -r requirements.txt && rm requirements.txt
# Install pytorch3d
ARG PYTORCH3D_VERSION="0.7.4"
RUN source ${HOME}/.zshrc && FORCE_CUDA=1 python -m pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v${PYTORCH3D_VERSION}"
# Install kaolin
ARG KAOLIN_VERSION="0.14.0"
RUN source ${HOME}/.zshrc && python -m pip install --no-cache-dir kaolin==${KAOLIN_VERSION} -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu117.html

# Set working directory
ARG WORK_DIR="code"
WORKDIR ${HOME}/${WORK_DIR}