# hash:sha256:e8b77d32cdd198d4096f0bdb216cf80c00bb7160119842af8b1618e8520162a3
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        keras==2.4.3 \
        matplotlib==3.5.1 \
        numpy==1.22.3 \
        pandas==1.4.2 \
        scikit-learn==1.0.2 \
        seaborn==0.12.2 \
        tensorflow==2.4.1 \
    && conda clean -ya
