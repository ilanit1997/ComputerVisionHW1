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
| 2 | Yolo5s        | 0         |  SGD    | Original+aug | 133      | 300   | 0.5     | 0.823       | 0.898        | 0.812       |
| 3 | Yolo5m        | 0         |  SGD    | Original+aug | 112      | 300   | 0.5     | 0.84       | 0.883        | 0.803       |
| 4 | YoloInit       | 0         |  SGD    | Original+aug | 53       | 300   | 0.5     | 0.5872       | 0.682        | 0.59       |
| 5 | Yolo5s        | 10        |  SGD    | Original+aug | 165      | 10   | 0.5     | 0.837       | 0.897        | 0.837       |
| 6 | Yolo5s        | 20         |  SGD    | Original+aug | 300      | 10   | 0.5     | 0.768       | 0.871        | 0.782       |
| 7 | Yolo5s        | 20         |  SGD    | Flipped+aug | 81      | 10   | 0.5     | 0.493       | 0.475        | 0.804       |
| 8 | Yolo5s        | 0         |  SGD    | Flipped+aug | 40      | 2   | 0.5     | 0.553       | 0.448        | 0.944       |
| 9 | Yolo5s        | 0         |  SGD    | Original+aug | 300      | 2   | 0.5     | 0.819       | 0.899        | 0.785       |
| 10 | Yolo5s        | 0         |  SGD    | Original+aug | 300      | 10   | 0.5     | 0.553       | 0.572        | 0.944       |
| 11 | Yolo5s        | 0         |  SGD    | Original+flipped +clean | 66      | 2   | 0.0     | 0.971       | 0.948        | 0.958       |
| 12 | Yolo5m        | 0         |  SGD    | Flipped+aug | 65      | 2   | 0.0     | 0.953       | 0.947        | 0.949       |


All the plots attached below were created by ClearML
1. ### Train:
Comparing 2 experiments: #11 and #5" 
<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/%2311%20VS%20%23%205%20-train%20loss.JPG" width="700" height="400">

Looking into train metrics, when in red we have # 5, and greens we have #11. It may seem that #5 is better in some metrics, but actually the differences are very small and #11 is much better is the cls loss. 

Comparing 2 experiments: #11 and #12: 

<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/%2311%20vs%20%2312%20-%20train%20loss.JPG" width="700" height="400">

Looking into the loss figures, we can see that both are very much close, but #11 is superior in regard to cls_loss, as it was also in previous figure

2. ### Validation:

## mAP of all experiments:
<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/%2311%20VS%20%23%205%20-valloss.JPG" width="700" height="400">


Comparing 2 experiments: #11 and #5:
<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/%2311%20VS%20%23%205%20-valloss.JPG" width="700" height="400">

Comparing 2 experiments: #11 and #12: 
#### Loss
<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/%2311%20vs%20%2312%20-%20validation%20loss.JPG" width="700" height="400">


#### mAP50 Metrics of # 1-10

<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/mAP50%20-%20val.png" width="700" height="400">

##### mAP Metrics per class # 11

<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/mAP%20results%20-%20validation%20%2311.JPG" width="700" height="400">

##### Metrics of # 11

<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/experimet%2311_val.png" width="700" height="400">

3. ### Test:
##### mAP25-75 of Test #11 per class
<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/mAP%20results%20-%20test%20%2311.JPG" width="700" height="400">

##### mAP25 of Test #11 per class + all
<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/mAP25%20results%20-%20test%20%2311.JPG" width="700" height="400">

##### mAP50 of Test #11 per class + all
<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/mAP50%20results%20-%20test%20%2311.JPG" width="700" height="400">

##### mAP75 of Test #11 per class + all
<img src="https://github.com/ilanit1997/ComputerVisionHW1/blob/master/results%20-%20plots/mAP75%20results%20-%20test%20%2311.JPG" width="700" height="400">





