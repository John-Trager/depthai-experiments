#!/usr/bin/env python3
'''
@author John Trager

road-segmentation-adas-0001 running on selected camera.
Run as:
python3 road_segmentation.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'
Blob created at http://luxonis.com:8080/ under openvino model zoo and selecting "road-segmentation-adas-0001"

road-segmentation-adas-0001
It can be treated as a four-channel feature map, where each channel is a probability of one of the classes: Back Ground, road, curb, mark.
The output is a blob with the shape [B, C=4, H=512, W=896].
'''

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import sys


cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='models/road-segmentation-adas-0001_12shave.blob', type=str)

args = parser.parse_args()

cam_source = args.cam_input
nn_path = args.nn_model

nn_shape_x = 896
nn_shape_y = 512

def decode_segmentation(output_tensor, class_colors=[[0,0,0],  [0,255,0]]):
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(nn_shape_y,nn_shape_x)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_segmentation(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.2,0)


# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_3)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam=None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(nn_shape_x,nn_shape_y)
    cam.setInterleaved(False)
    cam.preview.link(detection_nn.input)
elif cam_source == 'left':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif cam_source == 'right':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

if cam_source != 'rgb':
    manip = pipeline.createImageManip()
    manip.setResize(nn_shape_x,nn_shape_y)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

cam.setFps(30)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

start_time = time.time()
counter = 0
fps = 0
layer_info_printed = False
while True:
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_nn_input = q_nn_input.get()
    in_nn = q_nn.get()

    if in_nn_input is not None:
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, in_nn_input.getHeight(), in_nn_input.getWidth())
        frame = in_nn_input.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)


    if in_nn is not None:
        #print("NN received")
        layers = in_nn.getAllLayers()

        if not layer_info_printed:
            for layer_nr, layer in enumerate(layers):
                print(f"Layer {layer_nr}")
                print(f"Name: {layer.name}")
                print(f"Order: {layer.order}")
                print(f"dataType: {layer.dataType}")
                dims = layer.dims[::-1] # reverse dimensions
                print(f"dims: {dims}")
            #print('num layers: ',len(layers))
            layer_info_printed = True

        # get layer1 data
        layer1 = in_nn.getFirstLayerFp16()
        #print('fp16 layer 0: ',len(layer1))

        # reshape to numpy array
        dims = layer.dims[::-1]
        lay1 = np.asarray(layer1, dtype=np.float16).reshape(dims)

        # change from float16 to int32
        newLayer = np.array(lay1, dtype=np.int32)

        #split into separate object groups
        bg = newLayer[:,0,:,:]
        road = newLayer[:,1,:,:]
        curb = newLayer[:,2,:,:]
        mark = newLayer[:,3,:,:]

        #decode to visualization
        output_bg_colors = decode_segmentation(bg,class_colors=[[0,0,0],[0,0,255]]) #red
        output_road_colors = decode_segmentation(road,class_colors=[[0,0,0],[0,255,0]]) #green
        output_curb_colors = decode_segmentation(curb,class_colors=[[0,0,0],[255,0,0]]) #blue
        output_mark_colors = decode_segmentation(mark,class_colors=[[0,0,0],[255,0,255]]) #yellow


        if frame is not None:
            frame = show_segmentation(output_bg_colors, frame)
            frame = show_segmentation(output_road_colors, frame)
            frame = show_segmentation(output_curb_colors, frame)
            frame = show_segmentation(output_mark_colors, frame)

            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
            cv2.imshow("nn_input", frame)

    counter+=1
    if (time.time() - start_time) > 1 :
        fps = counter / (time.time() - start_time)

        counter = 0
        start_time = time.time()

    if cv2.waitKey(1) == ord('q'):
        break

