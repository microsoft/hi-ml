# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.187.0/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# Install Node.js
ARG NODE_VERSION="lts/*"
RUN su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"

# Install ncc to build format_coverage github action.
RUN npm install -g @vercel/ncc

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
COPY build_requirements.txt test_requirements.txt /tmp/conda-tmp/
RUN /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment to install a different version of Python than the default
# RUN conda install -y python=3.6 \
#     && pip install --no-cache-dir pipx \
#     && pipx reinstall-all

# install additional OS packages, i.e. make.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends make

# [Optional] Uncomment this section to hard code HIML test environment variables into image.
# ENV HIML_RESOURCE_GROUP="<< YOUR_HIML_RESOURCE_GROUP >>"
# ENV HIML_SUBSCRIPTION_ID="<< YOUR_HIML_SUBSCRIPTION_ID >>"
# ENV HIML_WORKSPACE_NAME="<< YOUR_HIML_WORKSPACE_NAME >>"
