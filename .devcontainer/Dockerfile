# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.194.3/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version: debian-11, debian-10, debian-9, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="buster"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

WORKDIR /home/root

RUN wget https://github.com/InsightSoftwareConsortium/ITK/archive/refs/tags/v5.2.0.tar.gz
RUN tar -xf v5.2.0.tar.gz
RUN mkdir -p /bin/ITK && cd /bin/ITK && cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DITK_USE_64BITS_IDS=ON -DITK_LEGACY_REMOVE=ON /home/root/ITK-5.2.0 && make