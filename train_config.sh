#!/bin/bash

## train
python train.py --project CV_course/CV_HW1 --name training_clean_1 --img 640 --batch 16 --epochs 3 --data clearml://f5742111fbf8422ebf1ce60212a989d0 --patience 100 --weights yolov5m.pt  --cache

## test:
## python yolov5/val.py --weights best.pt --task test --data cv_hw1_flipped_dataset-2/data.yaml