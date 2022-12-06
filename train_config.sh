#!/bin/bash

python  /home/student/HW1/yolov5/train.py --img  640 --batch  24 --save-period 3 --epochs 300 --data  clearml://92fb8e2c38014a168f42506f0bf7bc9d --cfg /home/student/HW1/yolov5/models/custom_yolov5s.yaml --weights '' --cache


## computed anchors: [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]