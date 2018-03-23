#!/bin/bash

python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /media/jiaren/ImageNet/SceneFlowData/ \
               --epochs 0 \
               --loadmodel ./trained/checkpoint_10.tar \
               --savemodel ./trained/



python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath /media/jiaren/ImageNet/data_scene_flow_2015/training/ \
                   --epochs 300 \
                   --loadmodel ./trained/checkpoint_10.tar \
                   --savemodel ./trained/

