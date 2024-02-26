xhost +local:docker
docker run -it --rm \
-e SDL_VIDEODRIVER=x11 \
-e DISPLAY=unix$DISPLAY \
--env='DISPLAY' \
--gpus all --privileged \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-v /media/server_storage/datasets:/workspace/datasets \
-v /media/server_storage/repositories/moon/pixel_error_recovery/:/workspace/pixel_error_recovery \
-v /dev/shm:/dev/shm \
-w /workspace/pixel_error_recovery \
manage.ketiauto.in/library/pixel_error_recovery_test:moon \
/bin/bash
xhost -local:docker