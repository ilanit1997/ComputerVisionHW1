import bbox_visualizer as bbv
import torch
import numpy as np
from helpers import *


## load paths of different files
PATH = 'sample_images/images'
FILES = os.listdir(PATH)
rand_idx = np.random.choice(list(range(len(FILES))), 1)[0]
curr_file = FILES[rand_idx]
image_path = os.path.join(PATH, curr_file) ## ex: P016_tissue1_6625
file_name = '_'.join(curr_file.split('_')[:2])
frame_idx = int(curr_file.split('_')[-1].split('.')[0])

## load ground truth files
ground_truth_df_left =  pd.read_csv(f'sample_videos/tools_left/{file_name}.txt', header=None, sep=' ', names=['start','end',  'label' ])
ground_truth_df_right = pd.read_csv(f'sample_videos/tools_right/{file_name}.txt',  header=None, sep=' ', names=['start','end',  'label'])
ground_truth_bbox = pd.read_csv(f'sample_images/bbox_labels/{file_name}_{frame_idx}.txt',  header=None, sep=' ',
                                names=["label_index", "xcenter", "ycenter", "w", "h"])
ground_truth_bbox = darknetbbox_to_yolo(ground_truth_bbox)


## load repo from original git yolov5
model = torch.hub.load('ultralytics/yolov5', 'custom',  path='best.pt', source='github')

## define model params
model.conf = 0.6 ## allow preds over this threshold
model.max_det = 2 ## predict max 2 classes
size= (640,640)

## load frame into cv
frame = cv2.imread(image_path)
frame = cv2.resize(frame, size)
frame_to_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

## inference
output = model(frame_to_rgb)
output.render()
output_df = output.pandas().xyxy[0]

try:
    if len(output_df) == 2:
        left_df, right_df = extract_left_right(output_df)
        boxes1 = [df_to_bbox(left_df), df_to_bbox(right_df)]
        labels = [left_df['name'].values[0], right_df['name'].values[0]]
    elif len(output_df) == 1:
        boxes1 = df_to_bbox(output_df)
        labels = [output_df['name'].values[0]]
except TypeError:
    # no detections
    boxes1 = []
    labels = []


frame = bbv.draw_multiple_rectangles(frame, boxes1, bbox_color=(255, 0, 0))
frame = bbv.add_multiple_labels(frame, labels, boxes1, text_bg_color=(255, 0, 0))

## Left
real_label_left = extract_label(ground_truth_df_left, frame_idx)
draw_text(frame, text=real_label_left, font_scale=1,pos=(500, 20),  text_color_bg=(255, 0, 0), draw='left')
## Right
real_label_right = extract_label(ground_truth_df_right, frame_idx)
draw_text(frame, text=real_label_right, font_scale=1, pos=(10, 20), text_color_bg=(255, 0, 0), draw='right')

## draw ground truth bbox
left_df, right_df = extract_left_right(ground_truth_bbox)
boxes1 = [df_to_bbox(left_df), df_to_bbox(right_df)]
labels = ['GT:' + left_df['name'].values[0], 'GT:' +  right_df['name'].values[0]]
frame = bbv.draw_multiple_rectangles(frame, boxes1)
frame = bbv.add_multiple_labels(frame, labels, boxes1, top = False)


# Display the resulting frame
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.imshow('Frame', frame)
cv2.waitKey(0) #is required so that the image doesn’t close immediately. It will Wait for a key press before closing the image.

# Closes all the frames
cv2.destroyAllWindows()


