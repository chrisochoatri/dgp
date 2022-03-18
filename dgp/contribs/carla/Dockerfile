# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.
FROM carlasim/carla:latest
WORKDIR /home/carla/Import
USER root
RUN apt-get update && apt-get install -y wget
USER carla
RUN wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.13.tar.gz
WORKDIR /home/carla
RUN ./ImportAssets.sh