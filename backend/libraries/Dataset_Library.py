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

 

for key, value in list(globals().items()):
    if callable(value):
        if "function" in str(type(value)):
            importedfuncs.append(key)
        
newmethods={'ClassificationPrepareYOLO': {'0': ['file', 'Dataset Path','Folder'], '1': ['text', 'Dataset Name'], '2': ['radio', 'Dataset Folder Structure', ['Labeled Images', 'Class Folders']], '3': ['text', 'Train Images Percentage (%)'], '4': ['text', 'Test Images Percentage (%)']},'ClassificationPrepareTorch': {'0': ['file', 'Dataset Path','Folder'], '1': ['text', 'Dataset Name'], '2': ['radio', 'Dataset Folder Structure', ['Labeled Images', 'Class Folders']], '3': ['text', 'Train Images Percentage (%)'], '4': ['text', 'Test Images Percentage (%)']},'YOLO_From_Label_Studio': {'0': ['file', 'Select Dataset Folder', 'Folder'], '1': ['text', 'Train Images Percentage (%)'], '2': ['text', 'Test Images Percentage (%)']}}
color={'ClassificationPrepareYOLO': '#32a852', 'ClassificationPrepareTorch': '#32a852','YOLO_From_Label_Studio': '#49b3c1'}









            

import os
import shutil
import random


def ClassificationPrepareYOLO(inps):
    def splitratio(c):
        splitdirs = ["train","test"]
        for i in splitdirs:
            os.mkdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}")
            for j in c:
                os.mkdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}/{j}")
        for i in c:
            imgs = os.listdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}")
            random.shuffle(imgs)
            for idx,j in enumerate(imgs):
                if (idx+1) <= (int(inps["vars"]["Train Images Percentage (%)"])/100)*len(imgs):
                    shutil.copyfile(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}/{j}", f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/train/{i}/{j}")
                else:
                    shutil.copyfile(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}/{j}", f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/test/{i}/{j}")
        for i in os.listdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}"):
            if i not in ["test","train"]:
                shutil.rmtree(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}")
        
    print(inps["vars"])
    dirs = os.listdir(inps["path"])
    classes = []
    if inps["vars"]["Dataset Folder Structure"]:
        for i in dirs:
            if (str(i).split(".")[0]).split("_")[0] not in classes:
                classes.append((str(i).split(".")[0]).split("_")[0])
        print(len(classes))
        try:
            os.mkdir("./DatasetsYOLO")
            print("created")
        except:
            print("not created")
            pass
        os.mkdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}")
        for i in classes:
            if len(classes)<10:
                os.mkdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}")
                for j in dirs:
                    if (str(j).split(".")[0]).split("_")[0] == i:
                        shutil.copyfile(f'{inps["path"]}/{j}', f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}/{j}")
        splitratio(classes)
    else:
        pass
        

    return {"Dataset Path":f"./DatasetsYOLO/{inps['vars']['Dataset Name']}","outimagenode":""}

def ClassificationPrepareTorch(inps):
    def splitratio(c):
        splitdirs = ["train","val"]
        for i in splitdirs:
            os.mkdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}")
            for j in c:
                os.mkdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}/{j}")
        for i in c:
            imgs = os.listdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}")
            random.shuffle(imgs)
            for idx,j in enumerate(imgs):
                if (idx+1) <= (int(inps["vars"]["Train Images Percentage (%)"])/100)*len(imgs):
                    shutil.copyfile(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}/{j}", f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/train/{i}/{j}")
                else:
                    shutil.copyfile(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}/{j}", f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/val/{i}/{j}")
        for i in os.listdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}"):
            if i not in ["val","train"]:
                shutil.rmtree(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}")
        
    print(inps["vars"])
    dirs = os.listdir(inps['vars']['Dataset Path'])
    classes = []
    if inps["vars"]["Dataset Folder Structure"]:
        for i in dirs:
            if (str(i).split(".")[0]).split("_")[0] not in classes:
                classes.append((str(i).split(".")[0]).split("_")[0])
        print(len(classes))
        try:
            os.mkdir("./DatasetsYOLO")
            print("created")
        except:
            print("not created")
            pass
        os.mkdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}")
        for i in classes:
            if len(classes)<10:
                os.mkdir(f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}")
                for j in dirs:
                    if (str(j).split(".")[0]).split("_")[0] == i:
                        shutil.copyfile(f'{inps["vars"]["Dataset Path"]}/{j}', f"./DatasetsYOLO/{inps['vars']['Dataset Name']}/{i}/{j}")
        splitratio(classes)
    else:
        pass
        

    return {"Dataset Path":f"./DatasetsYOLO/{inps['vars']['Dataset Name']}","outimagenode":""}







def YOLO_From_Label_Studio(inps):
    dataset = inps["vars"]["Select Dataset Folder"]
    with open(f"{dataset}/classes.txt","r") as file:  
        text = file.read()
    classes = text.split("\n")
    with open(f"{dataset}/config.yaml","w") as file:       
        content= f"path: {dataset}\ntrain: {dataset}/train\nval: {dataset}/val\n\n\nnames:\n"
        for idx,i in enumerate(classes):
            if i not in ["","\n"," "]:
                content+=f"  {idx}: {i}\n"
        file.write(content)
    images = os.listdir(f'{dataset}/images')
    random.shuffle(images)
    files = ["train","val","train/images","train/labels","val/images","val/labels"]
    try:
        for i in files:
            os.mkdir(f"{dataset}/{i}")
    except:
        shutil.rmtree(f'{dataset}/train')
        shutil.rmtree(f'{dataset}/val')
        for i in files:
            os.mkdir(f"{dataset}/{i}")
    for idx,j in enumerate(images):
        if (idx+1) <= (int(inps["vars"]["Train Images Percentage (%)"])/100)*len(images):
            shutil.copyfile(f"{dataset}/images/{j}", f"{dataset}/train/images/{j}")
            shutil.copyfile(f"{dataset}/labels/{str(j).rstrip('.jpg')}.txt", f"{dataset}/train/labels/{str(j).rstrip('.jpg')}.txt")
        else:
            shutil.copyfile(f"{dataset}/images/{j}", f"{dataset}/val/images/{j}")
            shutil.copyfile(f"{dataset}/labels/{str(j).rstrip('.jpg')}.txt", f"{dataset}/val/labels/{str(j).rstrip('.jpg')}.txt")
    return {"Dataset Path":f"{dataset}/config.yaml","outimagenode":inps["output_node"]} 

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
