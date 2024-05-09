import torchvision
from torchvision import transforms
import torch
from torch import no_grad
import requests
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# get the predictions of objects in the image
def get_predictions(pred, threshold=0.8, objects=None):
    
    predicted_classes= [(COCO_INSTANCE_CATEGORY_NAMES[i],p,[(box[0], box[1]), (box[2], box[3])]) for i,p,box in zip(list(pred[0]['labels'].numpy()),pred[0]['scores'].detach().numpy(),list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes=[  stuff  for stuff in predicted_classes  if stuff[1]>threshold ]
    
    if objects  and predicted_classes :
        predicted_classes=[ (name, p, box) for name, p, box in predicted_classes if name in  objects ]
    return predicted_classes
# draw the box around classified objects in the image
def draw_box(predicted_classes,image, threshold, rect_th= 10,text_size= 3,text_th=3):
    # Image preprocessing
    img=(np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8).copy()
    total_number_of_each_object={}
    # print(predicted_classes)

    # Draw the box around each predicted object
    for predicted_class in predicted_classes:
        # print(predicted_class)
        label=predicted_class[0]
        probability=predicted_class[1]
        box=predicted_class[2]
        tl = tuple(map(int, box[0]))  # Convert to tuple of integers
        br = tuple(map(int, box[1]))

        # Draw the rectangle
        cv2.rectangle(img, tl, br,(0, 255, 0), rect_th) 
        # Insert label text
        cv2.putText(img,label, tl,  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 
        cv2.putText(img, str(round(probability,2)), tl,  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,255),thickness=text_th)
        if label in total_number_of_each_object:
            total_number_of_each_object[label]+=1
        else:
            total_number_of_each_object[label]=1
    # plot the image
    plt.figure(figsize=(10,5))
    plt.title('Object Detection with threshold: '+str(threshold))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # plot the total number of each object
    plt.figure(figsize=(10,5))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(total_number_of_each_object)))
    plt.bar(total_number_of_each_object.keys(), total_number_of_each_object.values(), color=colors)
    # display the number on the bar
    for i, (key, value) in enumerate(total_number_of_each_object.items()):
        plt.text(i, value, value, ha='center', fontsize=14)
        plt.ylim(0, max(total_number_of_each_object.values()) + 5)
    plt.xlabel('Object')
    plt.ylabel('Total number')
    plt.title('Total number of each object with threshold: '+str(threshold))
    plt.show()

    # plt.figure(figsize=(10,5))
    # plt.bar(total_number_of_each_object.keys(), total_number_of_each_object.values())
    # # display the number on the bar
    # for i in range(len(total_number_of_each_object)):
    #     plt.text(i, list(total_number_of_each_object.values())[i], list(total_number_of_each_object.values())[i], ha = 'center')
    # plt.xlabel('Object')
    # plt.ylabel('Total number')
    # plt.title('Total number of each object')
    # plt.show()

    # free up memory
    del(img)
    del(image)

# Save the RAM
def save_RAM(image_=False):
    global image, img, pred
    torch.cuda.empty_cache()
    del(img)
    del(pred)
    if image_:
        image.close()
        del(image)

# Load the model
model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()

# set parameters to False
for name, param in model_.named_parameters():
    param.requires_grad = False

def model(x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat

# Labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'N/A', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'pigeon',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'fish',
    'shoe', 'handbag', 'sunglasses', 'hat', 'backpack', 'umbrella', 'watch'
]



# Find the total number of an object in the image above a certain threshold
def find_total_objects(image_path, object_name=None, threshold=0.8):
    # Load the image, preprocessing
    half = 0.5
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path)
    image.resize([int(half * s) for s in image.size])
    img = transform(image)
    # Find the objects in the image
    pred = model([img])
    pred_thresh=get_predictions(pred, threshold=0.8, objects=[object_name])
    draw_box(pred_thresh,img, threshold, rect_th=1, text_size=0.5, text_th=1)
    del pred_thresh
    save_RAM(image_=True)

def find_all_objects(image_path, threshold=0.8):
    # preprocess
    half = 0.5
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path)
    image.resize([int(half * s) for s in image.size])
    img = transform(image)
    pred = model([img])
    pred_thresh=get_predictions(pred, threshold=0.8)
    draw_box(pred_thresh,img, threshold, rect_th=1, text_size=0.5, text_th=1)
    del pred_thresh
    save_RAM(image_=True)
# Tests
# find_all_objects('t2.jpg', threshold=0.9)
# find_all_objects('t3.jpg', threshold=0.4)
