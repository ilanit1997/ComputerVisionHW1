# ComputerVisionHW1 

## Goal:
predict tool usage of of different test subjects, in a simulation of 2 procedures - tissue (=toilet paper), and ballon, sewing techniques.  

## model:
yolov5

## Results:
### Experiment Table

| # | Model type   | freeze    |  optimizer | Dataset type | epochs | maxdet | fliplr | Best mAP0.5 | Best precision | Best recall |
| :---: | :---:     | :---:     |  :---:     | :---:       | :---:   | :---: | :---:   | :---:       | :---:         |  :---:      |
| 1 | Yolo5s        | 0         |  Adam    | Original+aug | 34       | 300   | 0.5     | 0.201       | 0.6750        | 0.418       |


All the plots attached below were created by CleaML
1. Train:
Comparing 2 experiments: #11 and #5, where 
<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/%2311%20VS%20%23%205%20-train%20loss.JPG" width="700" height="400">



