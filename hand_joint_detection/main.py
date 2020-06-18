import numpy as np
import torch as T
import cv2
from utils import *
from dconv import *


load = False
epoch_count = 30
batch_size = 64
data_idx = 0

data_idx, epochs = create_epochs(epoch_count, batch_size, data_idx)
model = DConv(84, 42)

try:
    if load:
        model.load_checkpoint()
except:
    print("load failed")


i = 0
for epoch in epochs:
    i+=1
    epoch = prepare_epoch(epoch, 84)
    input, joints = get_data_for_pytorch_from_epoch(epoch)
    output = model(input)
    loss = model.learn(output, joints)
    model.save_checkpoint()

    print("------------------------------------")
    print("Epoch: ", i, "/",epoch_count, " Loss: ",loss)

print("Data idx = ", data_idx)
