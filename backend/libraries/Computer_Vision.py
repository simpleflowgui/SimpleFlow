import os, json,sys
sys.path.append('../Nodes.py')
from Nodes import writelog


info = {
    "lib_name":os.path.basename(__file__).split(".")[0],
    "funcs":[
    ]
}
methods = {

}

color = {}

importedfuncs = []


#Add your node functions here (function name is node name)
'''

example:

def node1():                    ---> node1 is node name
    print("node1")              ---> body of the function (function will print the string "node1")

'''

##funcs start

newmethods={'Input_Model': {'0': ['file', 'Model Path', 'File']}, 'Torch_Classify': {'0': ['file', 'Image Path', 'File'],'1':['select','Device',["cpu","cuda"]]}, 'Select_Model': {},'Train_Torch_Classify': {'0': ['file', 'Dataset Path', 'Folder'], '1': ['text', 'Epochs Number'],'2':['select','Device',["cpu","cuda"]]},'Torch_Detect': {'0': ['file', 'Image Path', 'File'],'1':['select','Device',["cpu","cuda"]]}, 'Train_Torch_Detect': {'0': ['file', 'Dataset Path', 'Folder'], '1': ['text', 'Epochs Number'],'2':['select','Device',["cpu","cuda"]]}, 'Camera_Video_Input': {},'Select_Model': {'0': ['radio', 'Task', ['Classification', 'Object Detection'], 'Select Model'], '1': ['select', 'Select Model', '{\n    "Classification":["AlexNet","ConvNeXt","DenseNet","EfficientNet","EfficientNetV2","GoogLeNet","Inception V3","MaxVit","MNASNet","MobileNet V2","MobileNet V3","RegNet","ResNet","ResNeXt","ShuffleNet V2","SqueezeNet","SwinTransformer","VGG","VisionTransformer","Wide ResNet"],\n        "Object Detection":["Faster R-CNN","FCOS","RetinaNet"]\n}', 'Select Builder'], '2': ['select', 'Select Builder', '{\n            "AlexNet":["alexnet"],\n            "ConvNeXt":["convnext_tiny","convnext_small","convnext_base","convnext_large"],\n            "DenseNet":["densenet121","densenet161","densenet169","densenet201"],\n            "EfficientNet":["efficientnet_b0","efficientnet_b1","efficientnet_b2","efficientnet_b3","efficientnet_b4","efficientnet_b5","efficientnet_b6","efficientnet_b7"],\n            "EfficientNetV2":["efficientnet_v2_s","efficientnet_v2_m","efficientnet_v2_l"],\n            "GoogLeNet":["googlenet"],\n            "Inception V3":["inception_v3"],\n            "MaxVit":["maxvit_t"],\n            "MNASNet":["mnasnet0_5","mnasnet0_75","mnasnet1_0","mnasnet1_3"],\n            "MobileNet V2":["mobilenet_v2"],\n            "MobileNet V3":["mobilenet_v3_large","mobilenet_v3_small"],\n            "RegNet":["regnet_y_400mf","regnet_y_800mf","regnet_y_1_6gf","regnet_y_3_2gf","regnet_y_8gf","regnet_y_16gf","regnet_y_32gf","regnet_y_128gf","regnet_x_400mf","regnet_x_800mf","regnet_x_1_6gf","regnet_x_3_2gf","regnet_x_8gf","regnet_x_16gf","regnet_x_32gf"],\n            "ResNet":["resnet18","resnet34","resnet50","resnet101","resnet152"],\n            "ResNeXt":["resnext50_32x4d","resnext101_32x8d","resnext101_64x4d"],\n            "ShuffleNet V2":["shufflenet_v2_x0_5","shufflenet_v2_x1_0","shufflenet_v2_x1_5","shufflenet_v2_x2_0"],\n            "SqueezeNet":["squeezenet1_0","squeezenet1_1"],\n            "SwinTransformer":["swin_t","swin_s","swin_b","swin_v2_t","swin_v2_s","swin_v2_b"],\n            "VGG":["vgg11","vgg11_bn","vgg13","vgg13_bn","vgg16","vgg16_bn","vgg19","vgg19_bn"],\n            "VisionTransformer":["vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"],\n            "Wide ResNet":["wide_resnet50_2","wide_resnet101_2"],\n            "MASK R-CNN":["maskrcnn_resnet50_fpn","maskrcnn_resnet50_fpn_v2"],\n            "Faster R-CNN":["fasterrcnn_resnet50_fpn","fasterrcnn_resnet50_fpn_v2","fasterrcnn_mobilenet_v3_large_fpn","fasterrcnn_mobilenet_v3_large_320_fpn"],\n            "FCOS":["fcos_resnet50_fpn"],\n            "RetinaNet":["retinanet_resnet50_fpn","retinanet_resnet50_fpn_v2"]\n}']},'Train_YOLO': {'0': ['radio', 'Mode', ['Classification', 'Detection', 'Segmentation'], ['Select Model','Dataset Path']], '1': ['select', 'Select Model', '{\n    "Classification": ["YOLOv8n-cls","YOLOv8s-cls","YOLOv8m-cls","YOLOv8l-cls","YOLOv8x-cls"],\n    "Detection": ["YOLOv8n","YOLOv8s","YOLOv8m","YOLOv8l","YOLOv8x"],\n    "Segmentation": ["YOLOv8n-seg","YOLOv8s-seg","YOLOv8m-seg","YOLOv8l-seg","YOLOv8x-seg"]\n}'], '2': ['file', 'Dataset Path', '{\n    "Classification": "Folder",\n    "Detection": "File",\n    "Segmentation": "File"\n}'], '3': ['text', 'Epochs'], '4': ['text', 'Image Size'], '5': ['text', 'Batch Size'],'6':['select','Device',["cpu","cuda"]]},'Predict_YOLO': {'0': ['radio', 'Model', ['Pre-trained', 'Custom']], '1': ['radio', 'Mode', ['Classification', 'Detection', 'Segmentation'], 'Select Model'], '2': ['select', 'Select Model', '{\n    "Classification": ["YOLOv8n-cls","YOLOv8s-cls","YOLOv8m-cls","YOLOv8l-cls","YOLOv8x-cls"],\n    "Detection": ["YOLOv8n","YOLOv8s","YOLOv8m","YOLOv8l","YOLOv8x"],\n    "Segmentation": ["YOLOv8n-seg","YOLOv8s-seg","YOLOv8m-seg","YOLOv8l-seg","YOLOv8x-seg"]\n}'], '3': ['file', 'Image Path', 'File']},
            #'Train_Torch_Instance_Segmentation': {'0': ['file', 'Dataset Path', 'Folder'], '1': ['select', 'Device', ['cpu', 'cuda']], '2': ['text', 'Epochs Number']}, 'Torch_Instance_Segmentation': {'0': ['file', 'Image Path', 'File']}, 
            'Segment_Anything': {'0': ['file', 'Image Path', 'File']}}
color={'Input_Model': '##0cb090ac660', 'Torch_Classify': '#33d16a', 'Select_Model': '##0cb090ac660','Train_Torch_Classify': '#33d16a','Torch_Detect': '#33d16a', 'Train_Torch_Detect': '#33d16a', 'Camera_Video_Input': '#33d16a','Select_Model': '#0cb090','Train_YOLO': '#0a6066','Predict_YOLO': '#33c7a2','Train_Torch_Instance_Segmentation': '#537efd',
        #'Torch_Instance_Segmentation': '#537efd', 
        'Segment_Anything': '#537efd','Select_Model': '#43c0ea'}



def cameraloop(inps,model,name):
    vidcap = cv2.VideoCapture(0)
    time.sleep(1)
    while True:
        #time.sleep(0.1)
        if vidcap.isOpened():
            ret, frame = vidcap.read()  #capture a frame from live video
            #check whether frame is successfully captured
            if ret:
                    cam = frame
            else:
                vidcap = cv2.VideoCapture(int(inps["prev_node"]["CameraVideoInput"]))
                time.sleep(2)
                print("Error : Failed to capture frame")
        # print error if the connection with camera is unsuccessful
        else:
            vidcap = cv2.VideoCapture(0)
            time.sleep(2)
            print("Cannot open camera")
        #results = model(str(inps["path"]))
        try:
            results = model(cam)
            res_plotted = results[0].plot()
            img = Image.fromarray(res_plotted)
            img = img.resize((512,512))
            b, g, r = img.split()
            img = Image.merge("RGB", (r, g, b))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            byte_im = buf.getvalue()
            byte_im = base64.b64encode(byte_im).decode('utf-8')
            byte_im_pil = byte_im
            byte_im = f"data:image/png;base64,{byte_im}"
            if name == "YOLO":
                if "cls" not in inps['vars']['Select Model']:
                    #Thread(target=cameraloop,args=(inps,model,vidcap,)).run()
                    #cameraloop(inps,model,vidcap)
                    for result in results:
                    #while True:
                        
                        log_string = ''
                        log_string_im = ''
                        log_string_im+=f"&{byte_im_pil}"
                        boxes = result.boxes
                        probs = result.probs
                        for box in boxes:  # there could be more than one detection
                            if len(box) == 0:
                                return log_string if probs is not None else f'{log_string}(no detections), '
                            if probs is not None:
                                n5 = min(len(result.names), 5)
                                top5i = probs.argsort(0, descending=True)[:n5].tolist()  # top 5 indices
                                log_string += f"{', '.join(f'{result.names[j]} {probs[j]:.2f}' for j in top5i)}, "
                                log_string_im += f"{', '.join(f'{result.names[j]} {probs[j]:.2f}' for j in top5i)}, "
                                log_string_im+="\n\n"
                                log_string+="\n\n"

                        clss = []
                        clsn = []
                        clsconf = []
                        if boxes:  
                            for box in boxes:
                                #print(clss.index[result.names[int(box.cls[0])]])
                                if result.names[int(box.cls[0])] not in clss:
                                    clsn.append(1)
                                    clsconf.append([format(box.conf[0],'.2f')])
                                else:
                                    clsn[clss.index(result.names[int(box.cls[0])])]+=1
                                    clsconf[clss.index(result.names[int(box.cls[0])])].append(format(box.conf[0],'.2f'))
                                if result.names[int(box.cls[0])] not in clss:
                                    clss.append(result.names[int(box.cls[0])])
                            print(clss)
                            for c in clss:                                                                       
                                    log_string += f"{clsn[clss.index(c)]} {c}{'s' * (clsn[clss.index(c)] > 1)} "
                                    for i in range(0,len(clsconf[clss.index(c)])):
                                        if i<len(clsconf[clss.index(c)])-1:
                                            log_string += f"{clsconf[clss.index(c)][i]}    "
                                        else:
                                            log_string += f"{clsconf[clss.index(c)][i]}    "
                                    log_string_im += f"&{clsn[clss.index(c)]} # {c}{'s' * (clsn[clss.index(c)] > 1)} # "
                                    for i in range(0,len(clsconf[clss.index(c)])):
                                        if i<len(clsconf[clss.index(c)])-1:
                                            log_string_im += f"{clsconf[clss.index(c)][i]}    "
                                        else:
                                            log_string_im += f"{clsconf[clss.index(c)][i]}    "

                        print(log_string)
                else:
                    log_string = f"{results[0].names[results[0].probs.top5[0]]}"
                    log_string_im = ""
                writelog(log_string)
            #Add real time camera for Torch
            else:
                if inps["prev_node"]["Torch"] == "Classify":
                    pass
                else:
                    pass
    
            # Data to be written
            dictionary = {
                "source":inps["output_node"],
                "data":byte_im
            }        
            # Serializing json
            json_object = json.dumps(dictionary, indent=4)
            
            # Writing to sample.json
            with open("../my-react-flow-app/sample.json", "w") as outfile:
                outfile.write(json_object)
        except:
            continue










for key, value in list(globals().items()):
    if callable(value):
        if "function" in str(type(value)):
            importedfuncs.append(key)
        




        

        




import re, io, base64, time
from PIL import Image, ImageOps
import os
import cv2
from typing import List, Tuple
import csv
from tqdm import tqdm
import wget
import warnings
import datetime
import torch, torchvision
from torchvision import datasets, models, transforms, utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
import ast
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import copy
import random
from xml.etree import ElementTree as et
import matplotlib.patches as patches
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import math
from ultralytics import YOLO
from threading import Thread
import subprocess
import tempfile
import importlib
import shutil
from importlib.machinery import SourcelessFileLoader
from Nodes import writelog
warnings.filterwarnings('ignore')

class CustDat(torch.utils.data.Dataset):
    def __init__(self , images , masks):
        self.imgs = images
        self.masks = masks
        
    def __getitem__(self , idx,path):
        images = sorted(os.listdir(f"{path}/Images"))
        masks = sorted(os.listdir(f"{path}/Masks"))
        img = Image.open(f"{path}/Images/PNGImages/" + images[idx]).convert("RGB")
        mask = Image.open(f"{path}/Masks" + masks[idx])
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = np.zeros((num_objs , mask.shape[0] , mask.shape[1]))
        for i in range(num_objs):
            masks[i][mask == i+1] = True
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin , ymin , xmax , ymax])
        boxes = torch.as_tensor(boxes , dtype = torch.float32)
        labels = torch.ones((num_objs,) , dtype = torch.int64)
        masks = torch.as_tensor(masks , dtype = torch.uint8)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        return T.ToTensor()(img) , target
    
    def __len__(self):
        return len(self.imgs)

for key, value in list(globals().items()):
        if callable(value):
            if "function" in str(type(value)):
                importedfuncs.append(key)

def Input_Model(inps):
    model = str(inps["vars"]["Model Path"])
    return {"model":model,"outimagenode":inps["output_node"]}

def Torch_Classify(inps):
    def pred_and_plot_image(inps,model: torch.nn.Module,
                            image_path: str, 
                            class_names: List[str],
                            image_size: Tuple[int, int] = (224, 224),
                            transform: torchvision.transforms = None,
                            image=None):

        print(f"The value of image: {type(image)}")
        writelog(f"The value of image: {type(image)}")
        # 2. Open image
        if image_path=="":
            img = Image.fromarray(image)
        else:
            img = Image.open(image_path).convert('RGB')
        
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))

        # 3. Create transformation for image (if one doesn't exist)
        if transform is not None:
            image_transform = transform
        else:
            image_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
            ])

        ### Predict on image ### 

        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        device = inps["vars"]["Device"]
        # 4. Make sure the model is on the target device
        model.to(device)

        # 5. Turn on model evaluation mode and inference mode
        model.eval()
        with torch.inference_mode():
            # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
            transformed_image = image_transform(img).unsqueeze(dim=0)

            # 7. Make a prediction on image with an extra dimension and send it to the target device
            target_image_pred = model(transformed_image.to(device))

        # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # 9. Convert prediction probabilities -> prediction labels
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

        # 10. Plot image with predicted label and probability

        plt.figure()
        plt.imshow(img)
        plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
        plt.axis(False)
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        byte_im = buf.getvalue()
        byte_im = base64.b64encode(byte_im).decode('utf-8')
        byte_im = f"data:image/png;base64,{byte_im}"
        print(inps["output_node"])
        writelog(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
        return {"data":byte_im,"outimagenode":inps["output_node"]}
    if "model" in inps["prev_node"]:
        endstring = re.search(".__.",os.path.basename(inps["prev_node"]["model"])).span()[0]
        transforms = torchvision.models.get_model_weights(os.path.basename(inps["prev_node"]["model"])[:endstring]).DEFAULT.transforms()
        classfile = re.search("_model",os.path.basename(inps["prev_node"]["model"])).span()[0]
        model = torch.load(inps["prev_node"]["model"])
        with open('./Models/Torch/'+os.path.basename(inps["prev_node"]["model"])[:classfile]+"_classes.txt","r") as f:
            data = f.read()
        data = data.split(",")
    else:
        model = torchvision.models.get_model(os.path.basename(inps["prev_node"]["builder_name"]), weights="DEFAULT")
        with open("./Models/Torch/ImageNet.txt","r") as f:
            data = f.read()
        d = ast.literal_eval(data)
        data = d.values()
        data  = list(data)
        transforms = torchvision.models.get_model_weights(os.path.basename(inps["prev_node"]["builder_name"])).DEFAULT.transforms()

    if "Image Path" in inps["vars"]:
        if inps["vars"]["Image Path"]=="()" or inps["vars"]["Image Path"]=="":
            vidcap = cv2.VideoCapture(int(inps["prev_node"]["id"]))
            if vidcap.isOpened():
                ret, frame = vidcap.read()  #capture a frame from live video
                #check whether frame is successfully captured
                if ret:
                        img = frame
                        imp = ""
                else:
                    print("Error : Failed to capture frame")
                    writelog("Error : Failed to capture frame")
            # print error if the connection with camera is unsuccessful
            else:
                print("Cannot open camera")
            #results = model(str(inps["path"]))
        else:
            img = None
            imp = str(inps["vars"]["Image Path"])
    else:
        vidcap = cv2.VideoCapture(int(inps["prev_node"]["id"]))
        if vidcap.isOpened():
            ret, frame = vidcap.read()  #capture a frame from live video
            #check whether frame is successfully captured
            if ret:
                    img = frame
                    imp = ""
            else:
                print("Error : Failed to capture frame")
                writelog("Error : Failed to capture frame")
        # print error if the connection with camera is unsuccessful
        else:
            print("Cannot open camera")
        #results = model(str(inps["path"]))
    return pred_and_plot_image(inps,model=model, 
        class_names = data,
        image_path=imp,
        transform=transforms, # optionally pass in a specified transform from our pretrained model weights
        image_size=(224, 224),
        image = img)


def Select_Model(inps):
    try:
        if "prev_node" in inps:
            DataSet = inps["prev_node"]["Dataset Path"] if "Dataset Path" in inps["prev_node"] else ""
        else:
            DataSet = ""
    except:
        DataSet = ""
    return {"model_name":inps["vars"]["Select Model"],"builder_name":inps["vars"]["Select Builder"],"outimagenode":inps["output_node"],"Dataset Path":DataSet}

def Train_Torch_Classify(inps):
    if "prev_node" in inps :
        if "Dataset Path" not in inps["prev_node"]:
            data_dir = inps["vars"]["Dataset Path"]
        elif inps["prev_node"]["Dataset Path"]!="":
            data_dir = inps["prev_node"]["Dataset Path"]
        else:
            data_dir = inps["vars"]["Dataset Path"]
    else:
        data_dir = inps["vars"]["Dataset Path"]
    data_transforms = {
        'train': torchvision.models.get_model_weights(inps["prev_node"]["builder_name"]).DEFAULT.transforms(),
        'val': torchvision.models.get_model_weights(inps["prev_node"]["builder_name"]).DEFAULT.transforms(),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                    shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    def train_model(model, criterion, optimizer, scheduler, num_epochs=int(inps["vars"]["Epochs Number"])):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            writelog(f'Epoch {epoch}/{num_epochs - 1}')
            writelog('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                writelog(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        writelog(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        writelog(f'Best val Acc: {best_acc:4f}')
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    device = inps["vars"]["Device"]
    model = torchvision.models.get_model(inps["prev_node"]["builder_name"], weights="DEFAULT").to(device)
    

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    try:
        for param in model.features.parameters():
            param.requires_grad = False
    except:
        pass
    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)
    
    try:
        layer = ""
        for i in range(len(model.classifier)-1,-1,-1):
            if re.search("Linear",str(model.classifier[i])):
                #print(model.classifier[i])
                layer = i
                model.classifier[layer] = nn.Linear(model.classifier[layer].in_features, output_shape).to(device)
                break
        if layer == "":
            for i in range(len(model.classifier)-1,-1,-1):
                if re.search("Conv",str(model.classifier[i])):
                    #print(model.classifier[i])
                    layer = i
                    model.classifier[layer] = nn.Conv2d(model.classifier[layer].in_features, output_shape).to(device)
                    break
    except:
        if re.search("swin",inps["builder_name"]):
            model.head = nn.Linear(model.head.in_features, output_shape).to(device)
        elif re.search("vit",inps["builder_name"]) :
            model.heads.head = nn.Linear(model.heads.head.in_features, output_shape).to(device)
        else:
            model.fc = nn.Linear(model.fc.in_features, output_shape).to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=int(inps["vars"]["Epochs Number"]))
    x = datetime.datetime.now()
    context_dict = {"model_path":model,"datetime":str(x.strftime("%d"))+"_"+str(x.strftime("%m"))+"_"+str(x.strftime("%y"))+"_"+str(x.strftime("%I"))+"_"+str(x.strftime("%M"))}
    torch.save(model,f'./Models/Torch/{inps["prev_node"]["builder_name"]}+".__."+{context_dict["datetime"]}+"_"+{os.path.basename(data_dir)}+"_model.pt"')
    classes = {}
    for i in range(0,len(class_names)):
                    classes[i] = class_names[i]         
    with open('./Models/Torch/'+str(inps["prev_node"]["builder_name"])+".__."+str(context_dict["datetime"])+"_"+str(os.path.basename(data_dir))+"_classes.txt", 'w') as file:
            for i in  range(0,len(class_names)): 
                if i < len(class_names)-1:
                    file.write(str(class_names[i])+",")
                else:
                    file.write(str(class_names[i]))
    return {"model":inps["prev_node"]["builder_name"]+".__."+context_dict["datetime"]+"_"+os.path.basename(data_dir)+'_model.pt',"outimagenode":inps["output_node"]}



class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, width, height,classes, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width

        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(files_dir))
                        if image[-4:]!='.xml']

        # classes: 0 index is reserved for background
        self.classes = classes

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color    
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        # annotation file
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)

            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)


            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height

            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:

            sample = self.transforms(image = img_res,
                                        bboxes = target['boxes'],
                                        labels = labels)

            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])



        return img_res, target

    def __len__(self):
        return len(self.imgs)



def Torch_Detect(inps): 
    if "model" in inps["prev_node"]:
        model = torch.load(inps["prev_node"]["model"],map_location=torch.device(inps["vars"]["Device"]))
        from torchvision import transforms as T
        transform = T.ToTensor()
        endstring = re.search(".__.",inps["prev_node"]["model"]).span()[0]
        classfile = re.search("_model",inps["prev_node"]["model"]).span()[0]
        with open('./Models/Torch/'+os.path.basename(inps["prev_node"]["model"])[:classfile]+"_classes.txt","r") as f:
            data = f.read()
        data = data.split(",")  
        if data[0] == "_":
            data = data[1:]
        #print(data)
    else:
        model = torchvision.models.get_model(inps["prev_node"]["builder_name"], weights="DEFAULT")
        transform = torchvision.models.get_model_weights(inps["prev_node"]["builder_name"]).DEFAULT.transforms()
        with open("./Models/Torch/Coco.txt","r") as f:
            data = f.read()
        data = data.split("\n")  
 


    def apply_nms(orig_prediction, iou_thresh=0.3):

        # torchvision returns the indices of the bboxes to keep
        keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]

        return final_prediction
    def plot_img_bbox(img, target):
        # plot the image and bboxes
        # Bounding boxes are defined as follows: x-min y-min width height
        fig, a = plt.subplots(1,1)
        fig.set_size_inches(5,5)
        a.imshow(img)
        labels = target['labels'].numpy()
        scores = target['scores'].numpy()
        for idx,box in enumerate((target['boxes'])):
            x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
            rect = patches.Rectangle((x, y),
                                        width, height,
                                        linewidth = 2,
                                        edgecolor = 'r',
                                        facecolor = 'none')

            # Draw the bounding box on top of the image
            a.add_patch(rect)
            plt.text(x, y-10,f"{data[labels[idx]-1]}\n{scores[idx]}",backgroundcolor= (1,0,0,0.5),color = "white")
        plt.axis('off')
        '''fig.canvas.draw()
        img = Image.frombytes('RGB',fig.canvas.get_width_height(),fig.canvas.tostring_rgb())'''
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        byte_im = buf.getvalue()
        byte_im = base64.b64encode(byte_im).decode('utf-8')
        byte_im = f"data:image/png;base64,{byte_im}"
        return {"data":byte_im,"outimagenode":inps["output_node"]}
    # function to convert a torchtensor back to PIL image
    def torch_to_pil(img):
        return torchtrans.ToPILImage()(img).convert('RGB')
    def get_transform(train):
        if train:
            return A.Compose([
                                A.HorizontalFlip(0.5),
                                ToTensorV2(p=1.0) 
                            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        else:
            return A.Compose([
                                ToTensorV2(p=1.0)
                            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})     
    
    
    
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    device = inps["vars"]["Device"]
    if "Image Path" in inps["vars"]:
        if inps["vars"]["Image Path"]=="()" or inps["vars"]["Image Path"]=="":
            vidcap = cv2.VideoCapture(int(inps["prev_node"]["id"]))
            if vidcap.isOpened():
                ret, frame = vidcap.read()  #capture a frame from live video
                #check whether frame is successfully captured
                if ret:
                        img = frame
                        img = Image.fromarray(img)
                        b, g, r = img.split()
                        img = Image.merge("RGB", (r, g, b))
                else:
                    print("Error : Failed to capture frame")
            # print error if the connection with camera is unsuccessful
            else:
                print("Cannot open camera")
            #results = model(str(inps["path"]))
        else:
            img = Image.open(str(inps["vars"]["Image Path"])).convert('RGB')
    else:
        vidcap = cv2.VideoCapture(int(inps["prev_node"]["id"]))
        if vidcap.isOpened():
            ret, frame = vidcap.read()  #capture a frame from live video
            #check whether frame is successfully captured
            if ret:
                    img = frame
                    img = Image.fromarray(img)
                    b, g, r = img.split()
                    img = Image.merge("RGB", (r, g, b))
            else:
                print("Error : Failed to capture frame")
        # print error if the connection with camera is unsuccessful
        else:
            print("Cannot open camera")
    # put the model in evaluation mode
    img = transform(img)
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])[0]    
    
    
    
    print('MODEL OUTPUT\n')
    nms_prediction = apply_nms(prediction, iou_thresh=0.01)
    return plot_img_bbox(torch_to_pil(img), nms_prediction)


def Train_Torch_Detect(inps):
    warnings.filterwarnings('ignore')
    # defining the files directory and testing directory
    data_dir = inps["vars"]["Dataset Path"]
    files_dir = data_dir+'/train' 
    test_dir = data_dir+'/test'
    # check dataset
    clss = []
    with open(f'{data_dir}/classes.txt','r') as f:
        data = f.read()
        print(data)
    data = data.split(",")
    for i in data:
        if re.search("\n",i):
            i = i[:re.search("\n",i).span()[0]]
        clss.append(i)
    data = clss
    print(data)
    dataset = ImagesDataset(files_dir, 224, 224,data)
            
    if re.search("fast",inps["prev_node"]["builder_name"]):
        def get_object_detection_model(num_classes):
            # load a model pre-trained pre-trained on COCO
            model = torchvision.models.get_model(inps["prev_node"]["builder_name"], weights="DEFAULT")
            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
            return model
    elif re.search("retinanet",inps["prev_node"]["builder_name"]):
        def get_object_detection_model(num_classes):
            model = torchvision.models.get_model(inps["prev_node"]["builder_name"], weights="DEFAULT")
            # replace classification layer 
            in_features = model.head.classification_head.conv[0][0].in_channels
            out_channels = model.head.classification_head.conv[0][0].out_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head.num_classes = num_classes
            cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
            torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
            torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
            # assign cls head to model
            model.head.classification_head.cls_logits = cls_logits
            return model            
    elif inps["prev_node"]["builder_name"] == "fcos_resnet50_fpn":
        def get_object_detection_model(num_classes):
            model = torchvision.models.get_model(inps["prev_node"]["builder_name"], weights="DEFAULT")
            # replace classification layer 
            in_features = model.head.classification_head.conv[9].in_channels
            out_channels = model.head.classification_head.conv[9].out_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head.num_classes = num_classes
            cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
            torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
            torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
            # assign cls head to model
            model.head.classification_head.cls_logits = cls_logits
            return model
    '''
    elif ctx["torch_model"] == "ssd300_vgg16":
        def get_object_detection_model(num_classes):

            model = torchvision.models.get_model(ctx["torch_model"], weights="DEFAULT")
            # replace classification layer 
            in_features = model.head.classification_head.conv[5].in_channels
            out_channels = model.head.classification_head.conv[5].out_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head.num_classes = num_classes

            cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
            torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
            torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
            # assign cls head to model
            model.head.classification_head.cls_logits = cls_logits

            return model
        
    elif ctx["torch_model"] == "ssdlite320_mobilenet_v3_large":
        def get_object_detection_model(num_classes):

            # load a model pre-trained pre-trained on COCO
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

            return model
        
    '''  
    
    
    
    def get_transform(train):
        if train:
            return A.Compose([
                                A.HorizontalFlip(0.5),
                                ToTensorV2(p=1.0) 
                            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        else:
            return A.Compose([
                                ToTensorV2(p=1.0)
                            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})          
                    

    # use our dataset and defined transformations
    dataset = ImagesDataset(files_dir, 480, 480,data, transforms= get_transform(train=True))
    dataset_test = ImagesDataset(files_dir, 480, 480,data, transforms= get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # train test split
    test_split = 0.2######################
    tsize = int(len(dataset)*test_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=4,########################
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=10, shuffle=False, num_workers=4,##########################
        collate_fn=utils.collate_fn)
    
    
    
    
    # to train on gpu if selected.
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = inps["vars"]["Device"]
    num_classes = len(data)
    # get the model using our helper function
    model = get_object_detection_model(num_classes)
    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)   
    
    
    
    # training for 10 epochs
    num_epochs = int(inps["vars"]["Epochs Number"]) ##########################################3

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        
        
    classes = {}
    for i in range(0,len(data)):
                    classes[i] = data[i]    
    x = datetime.datetime.now()
    context_dict = {"datetime":str(x.strftime("%d"))+"_"+str(x.strftime("%m"))+"_"+str(x.strftime("%y"))+"_"+str(x.strftime("%I"))+"_"+str(x.strftime("%M"))}
    torch.save(model,inps["prev_node"]["builder_name"]+".__."+context_dict["datetime"]+"_"+os.path.basename(data_dir)+'_model.pt')
    with open('./Models/Torch/'+str(inps["prev_node"]["builder_name"])+".__."+str(context_dict["datetime"])+"_"+str(os.path.basename(data_dir))+"_classes.txt", 'w') as file:
            for i in  range(0,len(data)): 
                if i < len(data)-1:
                    file.write(str(data[i])+",")
                else:
                    file.write(str(data[i]))
    return {"model":inps["prev_node"]["builder_name"]+".__."+context_dict["datetime"]+"_"+os.path.basename(data_dir)+'_model.pt',"outimagenode":inps["output_node"]}


def Camera_Video_Input(inps):
    return {"outimagenode":inps["output_node"]}

def Select_Model(inps):
    try:
        if "prev_node" in inps:
            DataSet = inps["prev_node"]["Dataset Path"] if inps["prev_node"]["Dataset Path"] else ""
        else:
            DataSet = ""
    except:
        DataSet = ""
    return {"model_name":inps["vars"]["Select Model"],"builder_name":inps["vars"]["Select Builder"],"outimagenode":inps["output_node"],"Dataset Path":DataSet}

def Train_YOLO(inps):
    if  "prev_node" not in inps  or "Dataset Path" not in inps["prev_node"]:
        data_dir = inps["vars"]["Dataset Path"]
    else:
        data_dir = inps["prev_node"]["Dataset Path"]
    try:
        model = YOLO(f"{inps['vars']['Select Model'].lower()}.pt")
    except: 
        wget.download(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{inps['model_name']}.pt")
        model = YOLO(f"{inps['vars']['Select Model'].lower()}.pt")
    context_dict = {"datetime":str(datetime.datetime.now())}
    sys.stderr = open("../my-react-flow-app/YOLOtrain.txt", "w")
    model.train(data=data_dir, epochs=int(inps["vars"]["Epochs"]), imgsz=int(inps["vars"]["Image Size"]),batch=int(inps["vars"]["Batch Size"]),name=inps["vars"]["Select Model"]+".__."+context_dict["datetime"]+"_"+os.path.basename(data_dir)+'_model.pt',device=inps["vars"]["Device"])
    return {"model":inps["vars"]["Select Model"]+".__."+context_dict["datetime"]+"_"+os.path.basename(data_dir)+'_model.pt',"outimagenode":inps["output_node"]}


def Predict_YOLO(inps):
    if inps["vars"]["Model"] == "Pre-trained":
        try:
            model = YOLO(f"{inps['vars']['Select Model'].lower()}.pt")
        except: 
            wget.download(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{inps['vars']['Select Model']}.pt")
            model = YOLO(f"{inps['vars']['Select Model'].lower()}.pt")
    else:
        model = YOLO(inps['model']) if "model" not in inps["prev_node"] else YOLO(inps['prev_node']['model'])
    if "CameraVideoInput" in inps["prev_node"]:
        cameraloop(inps,model,"YOLO")
    else:
        if "Image Path" in inps["vars"]:
            if inps["vars"]["Image Path"] == "()" or inps["vars"]["Image Path"] == "":
                vidcap = cv2.VideoCapture(int(inps["prev_node"]["id"]))
                if vidcap.isOpened():
                    ret, frame = vidcap.read()  #capture a frame from live video
                    #check whether frame is successfully captured
                    if ret:
                            cam = frame
                    else:
                        print("Error : Failed to capture frame")
                # print error if the connection with camera is unsuccessful
                else:
                    print("Cannot open camera")
                #results = model(str(inps["path"]))
                results = model(cam)
            else:
                img = Image.open(str(inps['vars']['Image Path'])).convert('RGB')
                results = model(img)
        else:
            vidcap = cv2.VideoCapture(int(inps["prev_node"]["id"]))
            if vidcap.isOpened():
                ret, frame = vidcap.read()  #capture a frame from live video
                #check whether frame is successfully captured
                if ret:
                        cam = frame
                else:
                    print("Error : Failed to capture frame")
            # print error if the connection with camera is unsuccessful
            else:
                print("Cannot open camera")
            #results = model(str(inps["path"]))
            results = model(cam)
        res_plotted = results[0].plot()
        img = Image.fromarray(res_plotted)
        img = img.resize((512,512))
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        byte_im = buf.getvalue()
        byte_im = base64.b64encode(byte_im).decode('utf-8')
        byte_im_pil = byte_im
        byte_im = f"data:image/png;base64,{byte_im}"
        if "cls" not in inps['vars']['Select Model']:
            #Thread(target=cameraloop,args=(inps,model,vidcap,)).run()
            #cameraloop(inps,model,vidcap)
            for result in results:
            #while True:
                
                log_string = ''
                log_string_im = ''
                log_string_im+=f"&{byte_im_pil}"
                boxes = result.boxes
                probs = result.probs
                for box in boxes:  # there could be more than one detection
                    if len(box) == 0:
                        return log_string if probs is not None else f'{log_string}(no detections), '
                    if probs is not None:
                        n5 = min(len(result.names), 5)
                        top5i = probs.argsort(0, descending=True)[:n5].tolist()  # top 5 indices
                        log_string += f"{', '.join(f'{result.names[j]} {probs[j]:.2f}' for j in top5i)}, "
                        log_string_im += f"{', '.join(f'{result.names[j]} {probs[j]:.2f}' for j in top5i)}, "
                        log_string_im+="\n\n"
                        log_string+="\n\n"

                clss = []
                clsn = []
                clsconf = []
                if boxes:  
                    for box in boxes:
                        #print(clss.index[result.names[int(box.cls[0])]])
                        if result.names[int(box.cls[0])] not in clss:
                            clsn.append(1)
                            clsconf.append([format(box.conf[0],'.2f')])
                        else:
                            clsn[clss.index(result.names[int(box.cls[0])])]+=1
                            clsconf[clss.index(result.names[int(box.cls[0])])].append(format(box.conf[0],'.2f'))
                        if result.names[int(box.cls[0])] not in clss:
                            clss.append(result.names[int(box.cls[0])])
                    print(clss)
                    for c in clss:                                                                       
                            log_string += f"{clsn[clss.index(c)]} {c}{'s' * (clsn[clss.index(c)] > 1)} "
                            for i in range(0,len(clsconf[clss.index(c)])):
                                if i<len(clsconf[clss.index(c)])-1:
                                    log_string += f"{clsconf[clss.index(c)][i]}    "
                                else:
                                    log_string += f"{clsconf[clss.index(c)][i]}    "
                            log_string_im += f"&{clsn[clss.index(c)]} # {c}{'s' * (clsn[clss.index(c)] > 1)} # "
                            for i in range(0,len(clsconf[clss.index(c)])):
                                if i<len(clsconf[clss.index(c)])-1:
                                    log_string_im += f"{clsconf[clss.index(c)][i]}    "
                                else:
                                    log_string_im += f"{clsconf[clss.index(c)][i]}    "

                print(log_string)
        else:
            log_string = f"{results[0].names[results[0].probs.top5[0]]}"
            log_string_im = ""
        writelog(log_string)
    return {"data":byte_im,"img":byte_im_pil,"log":log_string_im,"outimagenode":inps["output_node"]}


'''
def Train_Torch_Instance_Segmentation(inps):
    if "prev_node" not in inps or "Dataset Path" not in inps["prev_node"]:
        data_dir = inps["vars"]["Dataset Path"]
    else:
        data_dir = inps["prev_node"]["Dataset Path"]
    
    path = data_dir
    images = sorted(os.listdir(f"{path}/Images"))
    masks = sorted(os.listdir(f"{path}/Masks"))
    idx = 0
    img = Image.open(f"{path}/Images/" + images[idx]).convert("RGB")
    mask = Image.open(f"{path}/Masks/" + masks[idx])
    np.unique(mask)
    Image.fromarray(np.array(mask) == 1)
    Image.fromarray(np.array(mask) == 2)
    
    model = torchvision.models.get_model(inps["prev_node"]["builder_name"], weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features , 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , 2)
    
    transform = T.ToTensor()
    
    def custom_collate(data):
        return data


    images = sorted(os.listdir(f"{path}/Images"))
    masks = sorted(os.listdir(f"{path}/Masks"))
    num = int(0.9 * len(images))
    num = num if num % 2 == 0 else num + 1
    train_imgs_inds = np.random.choice(range(len(images)) , num , replace = False)
    val_imgs_inds = np.setdiff1d(range(len(images)) , train_imgs_inds)
    train_imgs = np.array(images)[train_imgs_inds]
    val_imgs = np.array(images)[val_imgs_inds]
    train_masks = np.array(masks)[train_imgs_inds]
    val_masks = np.array(masks)[val_imgs_inds]
    
    
    train_dl = torch.utils.data.DataLoader(CustDat(train_imgs , train_masks,path) , 
                                batch_size = 2 , 
                                shuffle = True , 
                                collate_fn = custom_collate , 
                                num_workers = 1 , 
                                pin_memory = True if torch.cuda.is_available() else False)
    val_dl = torch.utils.data.DataLoader(CustDat(val_imgs , val_masks,path) , 
                                batch_size = 2 , 
                                shuffle = True , 
                                collate_fn = custom_collate , 
                                num_workers = 1 , 
                                pin_memory = True if torch.cuda.is_available() else False)
    
    
    device = inps["vars"]["Device"]
    
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    all_train_losses = []
    all_val_losses = []
    flag = False
    epochs = inps["vars"]["Epochs Number"]
    for epoch in range(epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
        model.train()
        for i , dt in enumerate(train_dl):
            imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
            targ = [dt[0][1] , dt[1][1]]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
            loss = model(imgs , targets)
            if not flag:
                print(loss)
                flag = True
            losses = sum([l for l in loss.values()])
            train_epoch_loss += losses.cpu().detach().numpy()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        all_train_losses.append(train_epoch_loss)
        with torch.no_grad():
            for j , dt in enumerate(val_dl):
                imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
                targ = [dt[0][1] , dt[1][1]]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
                loss = model(imgs , targets)
                losses = sum([l for l in loss.values()])
                val_epoch_loss += losses.cpu().detach().numpy()
            all_val_losses.append(val_epoch_loss)
        print(epoch , "  " , train_epoch_loss , "  " , val_epoch_loss)
        
    x = datetime.datetime.now()
    context_dict = {"model_path":model,"datetime":str(x.strftime("%d"))+"_"+str(x.strftime("%m"))+"_"+str(x.strftime("%y"))+"_"+str(x.strftime("%I"))+"_"+str(x.strftime("%M"))}
    torch.save(model,f'../Models/Torch/{inps["prev_node"]["builder_name"]}+".__."+{context_dict["datetime"]}+"_"+{os.path.basename(data_dir)}+"_model.pt"')
    return {"model":inps["prev_node"]["builder_name"]+".__."+context_dict["datetime"]+"_"+os.path.basename(data_dir)+'_model.pt',"outimagenode":inps["output_node"]}

def Torch_Instance_Segmentation(inps):
    if "Image Path" in inps["vars"]:
        if inps["vars"]["Image Path"]=="()" or inps["vars"]["Image Path"]=="":
            vidcap = cv2.VideoCapture(0)
            if vidcap.isOpened():
                ret, frame = vidcap.read()  #capture a frame from live video
                #check whether frame is successfully captured
                if ret:
                        img = frame
                        imp = ""
                else:
                    print("Error : Failed to capture frame")
                    writelog("Error : Failed to capture frame")
            # print error if the connection with camera is unsuccessful
            else:
                print("Cannot open camera")
            #results = model(str(inps["path"]))
        else:
            img = None
            imp = str(inps["vars"]["Image Path"])
    else:
        vidcap = cv2.VideoCapture(0)
        if vidcap.isOpened():
            ret, frame = vidcap.read()  #capture a frame from live video
            #check whether frame is successfully captured
            if ret:
                    img = frame
                    imp = ""
            else:
                print("Error : Failed to capture frame")
                writelog("Error : Failed to capture frame")
        # print error if the connection with camera is unsuccessful
        else:
            print("Cannot open camera")
        #results = model(str(inps["path"]))
    device = "cpu"
    if "model" in inps["prev_node"]:
        model = torch.load(inps["prev_node"]["model"])
        model = torchvision.models.get_model(os.path.basename(inps["prev_node"]["builder_name"]), weights="DEFAULT")
    else:
        model = torchvision.models.get_model(inps["prev_node"]["builder_name"], weights="DEFAULT")
    model.eval()    
    if imp !="":
        img = Image.open(imp)
    transform = torchtrans.ToTensor()
    images = transform(img)
    with torch.no_grad():
        pred = model([images.to(device)])
    img = np.array(img)
    im2 = img.copy()
    for i in range(len(pred[0]['masks'])):
        msk = (pred[0]['masks'][i,0].cpu().detach().numpy() * 255).astype("uint8").squeeze()
        #plt.imshow(msk)
        scr = pred[0]['scores'][i].detach().cpu().numpy()
        if scr>0.8 :
            im2[:,:,0][msk>0.5] = random.randint(0,255)
            im2[:, :, 1][msk > 0.5] = random.randint(0,255)
            im2[:, :, 2][msk > 0.5] = random.randint(0, 255)
    plt.figure()
    plt.imshow(im2)
    plt.axis(False)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG')
    byte_im = buf.getvalue()
    byte_im = base64.b64encode(byte_im).decode('utf-8')
    byte_im = f"data:image/png;base64,{byte_im}"
    return {"data":byte_im,"outimagenode":inps["output_node"]}

'''
def Segment_Anything(inps):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        import urllib, wget
    except:            
        import pip
        def install(package):
            if hasattr(pip, 'main'):
                pip.main(['install', package])
            else:
                pip._internal.main(['install', package])
        install("git+https://github.com/facebookresearch/segment-anything.git")
        install("wget")
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        import urllib, wget
    
    if not os.path.exists("sam_vit_h_4b8939.pth"): 
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"  
        wget.download(url)
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    if "Image Path" in inps["vars"]:
        if inps["vars"]["Image Path"]=="()" or inps["vars"]["Image Path"]=="":
            vidcap = cv2.VideoCapture(0)
            if vidcap.isOpened():
                ret, frame = vidcap.read()  #capture a frame from live video
                #check whether frame is successfully captured
                if ret:
                        img = frame
                        imp = ""
                else:
                    print("Error : Failed to capture frame")
                    writelog("Error : Failed to capture frame")
            # print error if the connection with camera is unsuccessful
            else:
                print("Cannot open camera")
            #results = model(str(inps["path"]))
        else:
            img = None
            imp = str(inps["vars"]["Image Path"])
    else:
        vidcap = cv2.VideoCapture(0)
        if vidcap.isOpened():
            ret, frame = vidcap.read()  #capture a frame from live video
            #check whether frame is successfully captured
            if ret:
                    img = frame
                    imp = ""
            else:
                print("Error : Failed to capture frame")
                writelog("Error : Failed to capture frame")
        # print error if the connection with camera is unsuccessful
        else:
            print("Cannot open camera")
        #results = model(str(inps["path"]))
    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    if imp!="":
        img = cv2.imread(imp)
    else:
        img = np.array(img)
    img = cv2.resize(img,(10,10))
    masks = mask_generator.generate(img)
    plt.figure(figsize=(20,20))
    plt.imshow(img)
    show_anns(masks)
    plt.axis(False)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG')
    byte_im = buf.getvalue()
    byte_im = base64.b64encode(byte_im).decode('utf-8')
    byte_im = f"data:image/png;base64,{byte_im}"
    return {"data":byte_im,"outimagenode":inps["output_node"]}



##funcs end

#------------------------------------------------------------
#Add node parameter input methods (input file, input text,radio,dropdown)

def getdatamethod(newmethods):
    for i in newmethods:
        if i in methods:
            print(i)

            methods[i] = newmethods[i]
    info["methods"] = methods
    


def setinfo():
    l = []
    basefuncs = ["setinfo","exec","load","getdatamethod","value"]
    for key, value in list(globals().items()):
        if callable(value):
            if "function" in str(type(value)) and key not in importedfuncs:
                l.append(key)
    for i in l:
        if i not in basefuncs:
            arr = ["CustomNode",i]
            info["funcs"].append(arr)
            methods[i] = {}

def exec(nodename,inps):
    out = globals()[nodename](inps)
    json_object = json.dumps(out, indent=4)
    with open("./customnodeoutput.json","w") as file:
        file.write(json_object)

def load():
    global info
    info = {
        "lib_name":os.path.basename(__file__).split(".")[0],
        "color":color,
        "funcs":[
        ]
    }
    setinfo()
    getdatamethod(newmethods)
    return info
