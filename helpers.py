import cv2
import json
import os
import pandas as pd

tool_usage ={"T0": "no tool in hand" ,
 "T1":  "needle_driver",
  "T2": "forceps",
 "T3": "scissors"}


PATH = 'HW1_raw_dataset\\HW1_dataset\\tool_usage'

indexToYolo = {'0': 'Left_Empty', '1': 'Left_Forceps', '2': 'Left_Needle_driver', '3': 'Left_Scissors', '4': 'Right_Empty', \
          '5': 'Right_Forceps', '6' : 'Right_Needle_driver', '7': 'Right_Scissors'}


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



def extract_left_right(output_df):
    left_id, right_id = 1, 0
    if 'Left' in output_df['name'].values[0]:
        left_id, right_id = 0, 1

    left_df = pd.DataFrame(output_df.iloc[left_id, :]).T
    right_df = pd.DataFrame(output_df.iloc[right_id, :]).T

    return left_df, right_df

def df_to_bbox(df):
    # bbox = [xmin, ymin, xmax, ymax]
    bbox = [df['xmin'].values[0], df['ymin'].values[0], df['xmax'].values[0], df['ymax'].values[0]]
    bbox = [int(x) for x in bbox]
    return bbox


indexToYolo = {'0': 'Left_Empty', '1': 'Left_Forceps', '2': 'Left_Needle_driver', '3': 'Left_Scissors', '4': 'Right_Empty', \
          '5': 'Right_Forceps', '6' : 'Right_Needle_driver', '7': 'Right_Scissors'}

def darknetbbox_to_yolo(df, w_img=640, h_img=640):
    """
    xcenter = (xmin + w2) / w_img -- > xmin = xcenter*w_img - w2
    ycenter = (ymin + h2) / h_img -- > ymin = ycenter*h_img - h2
    w = w / w_img -- > w' = w*w_img
    h = h / h_img -- > h' = h*h_img
    xmax = xmin + w'
    ymax = ymin + h'
    """
    for i in range(len(df)):
        xcenter, ycenter, w, h = df.loc[i, 'xcenter'], df.loc[i, 'ycenter'], df.loc[i, 'w'], df.loc[i, 'h']
        wtag, htag = w*w_img, h*h_img
        df.loc[i, 'xmin'] = (xcenter - w /2) *w_img
        df.loc[i, 'xmax'] = df.loc[i, 'xmin'] + wtag
        df.loc[i, 'ymin'] = (ycenter - h /2) *h_img
        df.loc[i, 'ymax'] = df.loc[i, 'ymin'] + htag
        df.loc[i, 'name'] = indexToYolo[str(df.loc[i, 'label_index'])]
    return df
