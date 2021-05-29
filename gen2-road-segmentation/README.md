## [Gen2] road-segmentation-adas-0001 on DepthAI

This example show how to run the road-segmentation-adas-0001 model on DepthAI in the Gen2 API system.

TODO: [add gif or image for model in action]

## Pre-requisites
Have depthai (tested version 2.4) and opencv installed.<br>
To install dependencies:
```
python3 -m pip install depthai opencv-python
```

## Usage
```
python3 road_segmentation.py [-nn {path to model blob} -cam {which camera to use}]
```
Possible options for which camera to use: 'rgb', 'left', 'right'. <br>
There are two separate models included. One runs on 6 shave the other on 12. In testing there was no noticeable difference in performance.
