#!/bin/bash

docker --version
# docker pull
sudo docker pull nvcr.io/nvidia/l4t-ml:r32.6.1-py3
docker image ls


# docker exec -it <CONTAINER_ID_OR_NAME> bash


# Docker Run commands
# sudo docker run -it --rm --runtime nvidia --network host nvcr.io/nvidia/l4t-ml:r32.6.1-py3

# sudo docker run -it --rm --runtime nvidia --network host -v /home/user/project:/location/in/container nvcr.io/nvidia/l4t-ml:r32.6.1-py3