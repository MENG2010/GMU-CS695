#!/bin/bash

docker run --gpus all -it --mount source=$(pwd),target=/GMU-CS695/,type=bind -w /GMU-CS695 tensorflow/tensorflow:latest
