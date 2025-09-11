FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# Using a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore"

# Install git so we can clone the nnunet repository
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /opt/algorithm
RUN chown -R user /opt/algorithm
ENV PATH="/home/user/.local/bin:${PATH}"
USER user

COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

### Clone nnUNet
# Configure Git, clone the repository without checking out, then checkout the specific commit
# RUN git config --global advice.detachedHead false && \
# git clone https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/ 
COPY --chown=user:user ./algorithm/nnUNet/ /opt/algorithm/nnunet/
COPY --chown=user:user ./algorithm/medsam/ /opt/algorithm/medsam/
# Install a few dependencies that are not automatically installed
# RUN sudo apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip3 install \
        -e /opt/algorithm/medsam \
        -e /opt/algorithm/nnunet \
        graphviz \
        onnx \ 
        opencv-python-headless \
        SimpleITK && \
    rm -rf ~/.cache/pip
COPY --chown=user:user ./nnUNet_results/ /opt/algorithm/nnunet/nnUNet_results/

WORKDIR /opt/app

COPY --chown=user:user inference_v2.py /opt/app/
COPY --chown=user:user data_utils.py /opt/app/
RUN pip list | grep -E "torch|vision|timm"
### Set environment variable defaults
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT ["python", "inference_v2.py"]



# 使用带CUDA 11.8的Miniconda基础镜像
FROM continuumio/miniconda3:latest

# 显式声明CUDA版本（验证用）
ENV CUDA_VERSION=11.8
RUN nvcc --version || echo "CUDA not in PATH, will be handled by conda"

# 核心环境变量
ENV PYTHONUNBUFFERED=1 \
    CONDA_ENV_UMAMBA="umamba" \
    CONDA_ENV_NNUNET="nnunet" \
    SHARED_PYTHON_VERSION="3.10" \
    SHARED_TORCH_VERSION="2.0.1" \
    SHARED_TORCH_CUDA="cu118" \

# 安装CUDA工具链（如果基础镜像未包含）
RUN conda install -y -c nvidia \
    cuda-toolkit=${CUDA_VERSION} \
    cudnn=8.9.2 \
    nccl=2.18.1

# 创建Python 3.10环境
RUN conda create -n $CONDA_ENV_UMAMBA python=$SHARED_PYTHON_VERSION -y && \
    conda create -n $CONDA_ENV_NNUNET python=$SHARED_PYTHON_VERSION -y

# 在base环境安装指定CUDA版本的PyTorch
RUN conda install -y -c pytorch \
    pytorch==${SHARED_TORCH_VERSION} \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    pytorch-cuda=${CUDA_VERSION} \
    python=$SHARED_PYTHON_VERSION \
    --prefix /opt/conda

# 为两个环境创建软链接
RUN ln -s /opt/conda/lib/python${SHARED_PYTHON_VERSION}/site-packages/torch \
       /opt/conda/envs/$CONDA_ENV_UMAMBA/lib/python${SHARED_PYTHON_VERSION}/site-packages/ && \
    ln -s /opt/conda/lib/python${SHARED_PYTHON_VERSION}/site-packages/torchvision \
       /opt/conda/envs/$CONDA_ENV_UMAMBA/lib/python${SHARED_PYTHON_VERSION}/site-packages/ && \
    ln -s /opt/conda/lib/python${SHARED_PYTHON_VERSION}/site-packages/torch \
       /opt/conda/envs/$CONDA_ENV_NNUNET/lib/python${SHARED_PYTHON_VERSION}/site-packages/ && \
    ln -s /opt/conda/lib/python${SHARED_PYTHON_VERSION}/site-packages/torchvision \
       /opt/conda/envs/$CONDA_ENV_NNUNET/lib/python${SHARED_PYTHON_VERSION}/site-packages/

# 安装各自私有依赖
RUN /opt/conda/envs/$CONDA_ENV_UMAMBA/bin/pip install \
    /opt/algorithm/umamba/causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    /opt/algorithm/umamba/mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    -e /opt/algorithm/umamba/umamba
    /opt/conda/envs/$CONDA_ENV_NNUNET/bin/pip install \
    -e /opt/algorithm/nnunet \
    SimpleITK


# 验证环境
RUN echo "===== 环境验证 =====" && \
    /opt/conda/envs/$CONDA_ENV_UMAMBA/bin/python -c "import torch; \
    assert torch.__version__.startswith('$SHARED_TORCH_VERSION'), \
    f'U-Mamba Torch版本不符: {torch.__version__}'; \
    print(f'U-Mamba环境: PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    /opt/conda/envs/$CONDA_ENV_NNUNET/bin/python -c "import torch; \
    assert torch.version.cuda.startswith('$CUDA_VERSION'), \
    f'nnUNet CUDA版本不符: {torch.version.cuda}'; \
    print(f'nnUNet环境: PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"