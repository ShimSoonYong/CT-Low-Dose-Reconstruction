#!/bin/bash

docker build -t ai4mia .

docker container rm debug

docker run --gpus all -it --ipc=host \
	-p 6006:6006 \
	-v $(pwd):/workspace \
	--name debug \
	ai4mia
