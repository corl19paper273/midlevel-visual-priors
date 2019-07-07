docker run --runtime=nvidia -ti --rm \
    -v /tmp/.x11-unix:/tmp/.x11-unix \
    -v $PWD/runs/:/mnt/logdir/ \
    -v $PWD/configs/:/root/midlevel-reps/configs/ \
    -v $PWD/evkit/:/root/midlevel-reps/evkit/ \
    -v $PWD/scripts/:/root/midlevel-reps/scripts/ \
    -v $PWD/tnt/:/root/midlevel-reps/tnt/ \
    --network host --ipc=host \
    activeperception/habitat:1.0 bash
