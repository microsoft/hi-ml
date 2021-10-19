FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# install additional OS packages, i.e. make.
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    && apt-get -y install --no-install-recommends make \
    #
    # Install C compilers (gcc not enough, so just went with build-essential which admittedly might be overkill),
    # needed to build pandas C extensions
    && apt-get -y install build-essential \
    #
    # openslide c libs
    && apt-get -y install openslide-tools \
    #
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
