#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath dataset/ \
               --epochs 0 \
               --loadmodel ./trained/checkpoint_10.tar \
               --savemodel ./trained/



python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath dataset/data_scene_flow_2015/training/ \
                   --epochs 300 \
                   --loadmodel ./trained/checkpoint_10.tar \
                   --savemodel ./trained/

