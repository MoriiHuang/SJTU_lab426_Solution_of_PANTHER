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

### Clone nnUNet
# Configure Git, clone the repository without checking out, then checkout the specific commit
# RUN git config --global advice.detachedHead false && \
# git clone https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/ 
COPY --chown=user:user ./algorithm/U-Mamba/ /opt/algorithm/umamba/
# Install a few dependencies that are not automatically installed
# RUN sudo apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install\
    torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install\
    /opt/algorithm/umamba/causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl\
    /opt/algorithm/umamba/mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
USER root
RUN pip uninstall -y \
    numpy \
    opencv-python \
    opencv-python-headless 
USER user
RUN pip3 install \
        mrsegmentator \
        evalutils==0.4.2 \
        graphviz \
        onnx \ 
        opencv-python-headless \
        SimpleITK && \
    rm -rf ~/.cache/pip
RUN pip install -e /opt/algorithm/umamba/umamba
RUN pip install \
    transformers==4.30.0  numpy==1.26.4  opencv-python==4.9.0.80

RUN python -c "import torch; print(f'U-Mamba Torch version: {torch.__version__}')"
RUN python -c "import mamba_ssm; print(f'U-Mamba Timm version: {mamba_ssm.__version__}')"
COPY --chown=user:user ./nnUNet_results/ /opt/algorithm/nnunet/nnUNet_results/
WORKDIR /opt/app

USER root
RUN conda install -c conda-forge pydensecrf
USER user

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user inference_mam.py /opt/app/
COPY --chown=user:user data_utils.py /opt/app/
COPY --chown=user:user inference.sh /opt/app/
RUN pip list | grep -E "torch|vision|timm"
### Set environment variable defaults
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT ["bash", "inference.sh"]

