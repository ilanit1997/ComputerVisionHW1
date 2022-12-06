import cv2
import numpy as np
import bbox_visualizer as bbv
import torch
import pandas as pd


tool_usage ={"T0": "no tool in hand" ,
 "T1":  "needle_driver",
  "T2": "forceps",
 "T3": "scissors"}




def extract_label(df, frame_idx):
    start_fs, end_fs = df['start'].values, df['end'].values
    for i in range(len(df)):
        s, e = start_fs[i], end_fs[i]
        if frame_idx>s and frame_idx<e:
            tool = tool_usage[df['label'].values[i]]
            return tool
    return 'NO LABEL'


def draw_text(img, text,
          font=cv2.FONT_ITALIC,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0), draw='right'
          ):

    x, y = pos

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if draw == 'left' and len(text)> 10:
        x -= 120
        pos = (x,y)
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    else:
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)


    return text_size