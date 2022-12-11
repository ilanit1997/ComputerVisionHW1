import cv2
import numpy as np
import bbox_visualizer as bbv
import torch
from helpers import extract_label, draw_text
import pandas as pd
from smoothing import Smoothing
from evaluation import MetricEvaluator
import os

file_name = 'P026_tissue1'
cap = cv2.VideoCapture(f'videos/{file_name}.wmv')

ground_truth_df_left = pd.read_csv(f'tool_usage/tools_left/{file_name}.txt', header=None, sep=' ', names=['start', 'end', 'label' ])
ground_truth_df_right = pd.read_csv(f'tool_usage/tools_right/{file_name}.txt',  header=None, sep=' ', names=['start', 'end', 'label'])
#
smoother15_log = Smoothing(window_size=15, confidence_weighting_method="log", bbox_weighting_method=None)
smoother25_log = Smoothing(window_size=25, confidence_weighting_method="log", bbox_weighting_method=None)
smoother15_linear = Smoothing(window_size=15, confidence_weighting_method="linear", bbox_weighting_method=None)
smoother25_linear = Smoothing(window_size=25, confidence_weighting_method="linear", bbox_weighting_method=None)
smoother15_superlinear = Smoothing(window_size=15, confidence_weighting_method="super-linear", bbox_weighting_method=None)
smoother25_superlinear = Smoothing(window_size=25, confidence_weighting_method="super-linear", bbox_weighting_method=None)

smoothing_experiments = [smoother15_log, smoother25_log, smoother15_linear, smoother25_linear, smoother15_superlinear, smoother25_superlinear]
# tools_path = 'C:\\Users\\dovid\\PycharmProjects\\CV_hw1_old\\HW1_dataset\\HW1_dataset\\tool_usage'
evaluator = MetricEvaluator(ground_truth_df_left, ground_truth_df_right)

experiments = [(se, MetricEvaluator(ground_truth_df_left, ground_truth_df_right)) for se in smoothing_experiments]
experiments_dir = 'experiments'
if os.path.exists(experiments_dir) == False:
    os.mkdir(experiments_dir)


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

## load repo from original git yolov5
model = torch.hub.load('ultralytics/yolov5', 'custom',  path='best.pt', source='github')

model.conf = 0.6 ## allow preds over this threshold
model.max_det = 2 ## predict max 2 classes

#

i=0
# Read until video is completed
size= (640, 640)
while (cap.isOpened()):
    print(f"frame {i}")
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # prepare image for infer
        if i > 40:
            frame = cv2.resize(frame, size)
            frame_to_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ## infer
            output = model(frame_to_rgb)
            output.render()
            output_df = output.pandas().xyxy[0]
            outputs_smooth = smoother15_log.smooth(curr_output=output_df)
            # try:
            #     if len(outputs_smooth) == 2:
            #         right_df,  left_df = outputs_smooth[0], outputs_smooth[1]
            #         boxes1 = [left_df['bbox'], right_df['bbox']]
            #         labels = [left_df['prediction'], right_df['prediction']]
            #     elif len(outputs_smooth) == 1:
            #         df = outputs_smooth[0]
            #         boxes1 = [df['bbox']]
            #         labels = [df['prediction']]
            # except TypeError:
            #     ## TODO - handl
            #     boxes1 = []
            #     labels = []
            boxes1 = [output["bbox"] for output in outputs_smooth]
            labels = [output["prediction"] for output in outputs_smooth]

            # save predictions in evaluator
            [evaluator.convert_yolo_output_to_tool(label) for label in labels]
            # print current metrics / store metrics of current step in evaluator's memory
            # print(evaluator.calculate_recall())
            # print(evaluator.calculate_f1())
            # print(evaluator.calculate_precision())
            # print(evaluator.calculate_accuracy())
            # print(evaluator.calculate_macro_f1_score())
            # print()
            # print()
            evaluator.calculate_all_metrics()

            # experiments
            for se, ee in experiments:
                smoothed = se.smooth(curr_output=output_df)
                smoothed_labels = [output["prediction"] for output in smoothed]
                [ee.convert_yolo_output_to_tool(label) for label in smoothed_labels]
                ee.calculate_all_metrics()
                ee.history_to_pickle(experiments_dir + '/' + file_name + se.smoother_params)


            frame = bbv.draw_multiple_rectangles(frame, boxes1, bbox_color=(255, 0, 0))
            frame = bbv.add_multiple_labels(frame, labels, boxes1, text_bg_color=(255, 0, 0))

            ## Left
            real_label_left = extract_label(ground_truth_df_left, i)
            draw_text(frame, text=real_label_left, font_scale=1,pos=(500, 20),  text_color_bg=(255, 0, 0), draw='left')
            ## Right
            real_label_right = extract_label(ground_truth_df_right, i)
            draw_text(frame, text=real_label_right, font_scale=1, pos=(10, 20), text_color_bg=(255, 0, 0), draw='right')

            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            last_frame = frame.copy()

            evaluator.history_to_pickle("metric_evaluation_test")

            # Press Q on keyboard to  exit
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

        # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()