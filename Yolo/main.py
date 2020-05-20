import time
import sys
import torch as T
import numpy as np
import cv2
from utils import *
from darknet import *


input_name = "input.m4v"
output_name = "output.avi"

num_classes = 80
classes = load_classes("data/coco.names")
colors = create_colors(num_classes)
in_dim = 416

CUDA = T.cuda.is_available()

yolo = Darknet("config.cfg","weights/yolov3.weights")

if CUDA:
    yolo.cuda()

yolo.eval()

cap = cv2.VideoCapture(input_name)

if not cap.isOpened():
    print("Coudnt open source. Terminating...")
    sys.exit()
else:
    print("Video opened")

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
ret, img = cap.read()
shape = img.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_name, fourcc, fps, (shape[1], shape[0]))

print("Process began...")
start = time.time()
frame_start = time.time()
done_frames = 0
result = 0


while 1:
    object_detected = True
    ret, frame = cap.read()

    if ret is False:
        break

    if done_frames % 3 == 0:
        img = prep_image(frame.copy() ,in_dim)
        result = yolo(img, CUDA)
        result = adjust_results(result, 0.5, num_classes)

    try:
        write_result(result.clone(), frame, in_dim, classes, colors)
    except:
        object_detected = False

    out.write(frame)

    frame_end = time.time()

    print("----------------------\n")
    print("Frame is done")
    print("Frame took: ", frame_end - frame_start, " secs")
    print("Progress: %",done_frames*100/frames)
    if not object_detected:
        print("\nNO object detected in this frame...")
    print("\n----------------------")

    frame_start = frame_end
    done_frames += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


end = (time.time() - start)/60
print("----------------------\n")
print("Video is done")
print("Fps of video: ", fps)
print("The process took: ", end, " mins")
print("Cuda: ", CUDA)
print("\n----------------------")

cap.release()
out.release()
cv2.destroyAllWindows()
