# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.

docker-build-carla: 
	docker build --pull -f Dockerfile -t carla:latest .

docker-run-carla:
	docker run --privileged --gpus all --net=host  -e DISPLAY=:0  -v /tmp/.X11-unix:/tmp/.X11-unix:rw carla:latest /bin/bash ./CarlaUE4.sh -vulkan -RenderOffScreen -nosound -prefernvidia -quality-level=Epic
