import cv2
import numpy as np
import torch as T
from torch.autograd import Variable


"""
 returns list of (x, y)
"""
def read_joints(joint_path):
    file = open(joint_path, "r")
    lines = file.readlines()
    joints = []

    for line in lines:
        l = line.split()
        x = int(float(l[1]))
        y = int(float(l[2]))
        joints.append([x, y])

    file.close()
    return joints


"""
 returns (x1, y1) (x2, y2) -> top-left, bottom-right
"""
def read_bbox(bbox_path):
    file = open(bbox_path, "r")
    lines = file.readlines()
    bbox = []

    for line in lines:
        l = line.split()
        bbox.append(int(l[1]))

    p1 = [bbox[1], bbox[0]]
    p2 = [bbox[3], bbox[2]]

    file.close()
    return p1, p2


"""
 1 epoch =  {images, bboxs, joınts} x batch_size

 Note: Data_1 things must change if you need to use different
 part of the dataset
"""
def create_one_epoch(batch_size, data_idx):
    epoch = []

    for i in range(int(batch_size/4)):
        for j in range(1,5):
            img_path = "frames/data_1/" + str(data_idx) + "_webcam_" + str(j) + ".jpg"
            bbox_path = "bboxs/data_1/" + str(data_idx) + "_bbox_" + str(j) + ".txt"
            joint_path = "joints/data_1/" + str(data_idx) + "_jointsCam_" + str(j) + ".txt"

            # For debug
            print(img_path)
            print(data_idx)

            batch = {}

            try:
                img = cv2.imread(img_path)
                joints = read_joints(joint_path)
                bbox = read_bbox(bbox_path)
            except:
                print("Data load failed...")
                break

            batch["image"] = img
            batch["joints"] = joints
            batch["bbox"] = bbox

            epoch.append(batch)

        data_idx += 1

    return data_idx, epoch



"""
 return list of epochs
 1 epoch =  {image, bbox, joınt} x batch_size
 data_idx: special variable to navigate in dataset
"""
def create_epochs(epoch_count, batch_size=64, data_idx=0):
    epochs = []

    for i in range(epoch_count):
        data_idx, epoch = create_one_epoch(batch_size, data_idx)
        epochs.append(epoch)

    return data_idx, epochs


def pad_image(img, inp_dim=84):
    w, h = img.shape[1], img.shape[0]

    resize_rate = inp_dim/max(w, h)
    new_w = int(w * resize_rate)
    new_h = int(h * resize_rate)

    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.zeros((inp_dim, inp_dim, 3), dtype=np.uint8)
    canvas[(inp_dim-new_h)//2:(inp_dim+new_h)//2, (inp_dim-new_w)//2:(inp_dim+new_w)//2 ,:] = resized_image

    return canvas


def crop_image(img, bbox):
    x = bbox[0][0]
    y = bbox[0][1]
    x1 = bbox[1][0]
    y1 = bbox[1][1]

    img = img[y:y1,x:x1,:]
    return img


def prepare_epoch(epoch, inp_dim=84):
    new_epoch = []
    for batch in epoch:
        batch = prepare_batch(batch, inp_dim)
        new_epoch.append(batch)
    return new_epoch


def prepare_batch(batch, inp_dim):
    img = batch["image"]
    joints = batch["joints"]
    bbox = batch["bbox"]

    img = crop_image(img, bbox)

    scale = inp_dim/max(img.shape)
    offsetX = int((inp_dim - scale*img.shape[1]) / 2)
    offsetY = int((inp_dim - scale*img.shape[0]) / 2)

    for j in joints:
        j[0] -= bbox[0][0]
        j[1] -= bbox[0][1]
        j[0] = int((j[0]*scale) + offsetX)
        j[1] = int((j[1]*scale) + offsetY)

    img = pad_image(img, inp_dim)
    #img = cv2.color(img, cv2.COLOR_RGB2GRAY)

    batch["image"] = (img)
    batch["joints"] = joints

    return batch


def get_data_for_pytorch_from_epoch(epoch):
    imgs = []
    joints = []

    for batch in epoch:
        img = batch["image"]
        img = img[:,:,::-1].transpose((2,0,1)).copy()
        img = T.from_numpy(img).float().div(255.0)
        imgs.append(img)

        # remove comment to normalize joint coordinates
        # and change activation to sigmoid in neural net
        j = T.tensor(batch["joints"]).view(-1).float()#.div(img.size()[1])
        joints.append(j)

    imgs = Variable(T.stack(imgs))
    joints = Variable(T.stack(joints))

    return imgs, joints


def visuilize_batch(batch):
    img = batch["image"]
    joints = batch["joints"]

    for j in joints:
        cv2.circle(img, (j[0],j[1]), 2, (255,0,0))

    img = cv2.resize(img, (500,500), interpolation = cv2.INTER_CUBIC)
    cv2.imshow("frame", img)
    cv2.waitKey(5000)
