import re, io, base64
from PIL import Image, ImageOps
import os
import cv2
import json
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import importlib


def foo(inps):
    return "foo"

def Measure(inps):
    writelog(inps)
    return "Measure"


def Measure2(inps):
    writelog("Measure2")
    return "Measure2"


def End(inps):
    return "End"


def OutImage(inps):
    return "OutImage"

def Loop(inps):
    return {"Loop":"Loop","outimagenode":inps["output_node"]}


def ImageResize(inps):
    if "path" in inps and inps["path"] !="()":
        imp = str(inps["path"])
    else:
        imp = inps["prev_node"]["img"]
        imp = str(imp).encode('utf-8')
        imp = io.BytesIO(base64.b64decode(imp))
    im = Image.open(imp)
    im_resize = im.resize((500, 500))
    buf = io.BytesIO()
    im_resize.save(buf, format='PNG')
    byte_im = buf.getvalue()
    byte_im_b = base64.b64encode(byte_im).decode('utf-8')
    byte_im = f"data:image/png;base64,{byte_im_b}"
    return {"data":byte_im,"img":byte_im_b,"outimagenode":inps["output_node"]}


def ImageFlip(inps):
    if "path" in inps and inps["path"] !="()":
        imp = str(inps["path"])
    else:
        imp = inps["prev_node"]["img"]
        imp = str(imp).encode('utf-8')
        imp = io.BytesIO(base64.b64decode(imp))
    im = Image.open(imp)
    im_flipped = ImageOps.mirror(im)
    buf = io.BytesIO()
    im_flipped.save(buf, format='PNG')
    byte_im = buf.getvalue()
    byte_im_b = base64.b64encode(byte_im).decode('utf-8')
    byte_im = f"data:image/png;base64,{byte_im_b}"
    return {"data":byte_im,"img":byte_im_b,"outimagenode":inps["output_node"]}


def CameraVideoInput(inps):
    return {"CameraVideoInput":inps["camindex"],"outimagenode":inps["output_node"],"id":inps["camindex"]}

def SelectCamera(inps):
    return {"id":inps["camindex"],"outimagenode":inps["output_node"]}

def CustomNode(inps):
    spec = importlib.util.spec_from_file_location(inps["lib"], f"./libraries/{inps['lib']}.py")
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    foo.exec(inps["custom_node"],inps)
    with open("customnodeoutput.json","r") as file:
        out = json.load(file)   
    return out












def upload(node,inps):
    root = tk.Tk()
    root.withdraw()
    if "upload_type" not in inps or str(inps["upload_type"]).lower()=="file":
        file_path = filedialog.askopenfilename()
    else:
        file_path = filedialog.askdirectory()
    root.destroy()
    return {"path":str(file_path),"idx":int(str(node).split("_")[-1]),"path_shortcut":str(file_path).split("/")[-1]}


def saveflow(flow,filename):
    json_object = json.dumps(flow, indent=4)
    with open(f"../my-react-flow-app/{str(filename)}.json", "w") as outfile:
        outfile.write(json_object)
    return 0

def writelog(log):
    with open("../my-react-flow-app/test.txt", "a") as outfile:
        outfile.write(log+"\n")

def clearlog():
    with open("../my-react-flow-app/test.txt", "w") as outfile:
        outfile.write("")
    return {"outimagenode":""}


def getflowslist():
    files = os.listdir("../my-react-flow-app")
    flowslist = []
    nameslist = []
    for f in files:
        if re.search(".json",os.path.basename(f)):
            flowslist.append(os.path.basename(f))
            nameslist.append(os.path.basename(f).split(".")[0])
    return{"outimagenode":"","flowslist":flowslist,"nameslist":nameslist}


libdict = {}
def loadnodes():
    libdict = {}
    libs = os.listdir("./libraries")
    for idx,lib in enumerate(libs):
        if os.path.isfile(f"./libraries/{lib}") and not os.path.isdir(f"./libraries/{lib}"):
            spec = importlib.util.spec_from_file_location(lib.split('.')[0], f"./libraries/{lib.split('.')[0]}.py")
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            libdict[idx] = foo.load()
    newlibdict = {}
    for idx,i in enumerate(libdict):
        newlibdict[idx] = libdict[i]
    return newlibdict


def savelibrary(lib,imports,text,inputmethods,colors):
    libs = os.listdir("libraries")
    if f"{lib}.py" in libs:
        with open(f"libraries/{lib}.py","r") as file:
            content = file.read()
        pattern1 = "##funcs start"
        pattern2 = "##funcs end"
        pattern3 = "newmethods={.*}"
        pattern4 = "color={.*}"
        flag1 = re.search(pattern1,content).span()[1]
        flag2 = re.search(pattern2,content).span()[0]
        newcontent = content[flag1:flag2]
        flag3 = re.search(pattern3,newcontent).span()
        flag4 = re.search("color=\{.*\}",newcontent).span()
        end = flag3[1]
        while content[end+1] == "}":
            end+=1
        end2 = flag4[1]
        while content[end2+1] == "}":
            end2+=1
        oldcolors = newcontent[flag4[0]+6:end2-1]
        for color in colors:
            if re.search(color,oldcolors):
                pat = re.search(color,oldcolors).span()[1]+5
                oldcolors = re.sub(oldcolors[pat:re.search(color,oldcolors).span()[1]+re.search("'",oldcolors[pat:]).span()[0]],colors[color],oldcolors)
        oldmethods = newcontent[flag3[0]+11:end-1]
        for method in inputmethods:
            if re.search(f"^{method}$",oldmethods):
                pat = re.search(method,oldmethods).span()[1]+3
                oldmethods = re.sub(oldmethods[pat:re.search(method,oldmethods).span()[1]+re.search("\}",oldmethods[pat:]).span()[0]],str(inputmethods[method]),oldmethods)
        newcontent = newcontent[:flag3[0]]+newcontent[end+1:]+"\n"
        newcontent = re.sub("color=\{.*\}","",newcontent)
        with open(f"./customheader.txt","r") as file:
            header = file.read()
        with open(f"./customfooter.txt","r") as file:
            footer = file.read()
        with open(f"libraries/{lib}.py","w") as file:
            file.write(header+"\n")
        newstring = '''
for key, value in list(globals().items()):
    if callable(value):
        if "function" in str(type(value)):
            importedfuncs.append(key)
        '''
        if imports != "":
            with open(f"libraries/{lib}.py","a") as file:
                file.write(imports+"\n"+newstring+"\n")
        with open(f"libraries/{lib}.py","a") as file:
            if oldmethods !="":
                file.write("newmethods="+str(oldmethods)+","+str(inputmethods)[1:]+"\n")
            else:
                file.write("newmethods="+str(inputmethods)[1:]+"\n")
        with open(f"libraries/{lib}.py","a") as file:
            if oldcolors != "":
                file.write("color="+str(oldcolors)+","+str(colors)[1:]+"\n")
            else:
                file.write("color="+str(colors)[1:]+"\n")
        with open(f"libraries/{lib}.py","a") as file:
            file.write(newcontent+text+"\n"+footer)
    else:
        with open(f"./customheader.txt","r") as file:
            header = file.read()
        with open(f"./customfooter.txt","r") as file:
            footer = file.read()
        with open(f"libraries/{lib}.py","w") as file:
            file.write(header+"\n")
        newstring = '''
for key, value in list(globals().items()):
    if callable(value):
        if "function" in str(type(value)):
            importedfuncs.append(key)
        '''
        if imports != "":
            with open(f"libraries/{lib}.py","a") as file:
                file.write(imports+"\n"+newstring+"\n")
        else:
            with open(f"libraries/{lib}.py","a") as file:
                file.write(newstring+"\n")
        with open(f"libraries/{lib}.py","a") as file:
            file.write("newmethods="+str(inputmethods)+"\n")
        with open(f"libraries/{lib}.py","a") as file:
            file.write("color="+str(colors)+"\n")
        with open(f"libraries/{lib}.py","a") as file:
            file.write(text+"\n"+footer)
            #pattern = '"color":\{\},'
    return {"outimagenode":""}


def loadcams():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    if len(arr)>1:
        cameras = ",".join(arr)
    elif len(arr) == 1:
        cameras = str(arr[0])
    else:
        cameras = ""
    return {"camslist":cameras}


        





functs = {
    "Start":foo,
    "Measure":Measure,
    "Measure2":Measure2,
    "End":End,
    "ImageResize":ImageResize,
    "OutImage":OutImage,
    "ImageFlip":ImageFlip,
    "CameraVideoInput":CameraVideoInput,
    "SelectCamera":SelectCamera,
    "CustomNode":CustomNode,
    "Loop":Loop
}



outs = {}
def mainfunc(edges,inps,loopout,lf):
    try:
        with open("./processpid.txt","w") as file:
            file.write(str(os.getpid()))
        outs = {}
        que = []
        nodes_with_out = {}
        if lf=="":
            t = "start"
            for j in range(0,len(edges)):
                for i,x in enumerate(edges):
                    if re.search(t,x["source"]):
                        if not re.search("OutImage",x["target"]):
                            que.append(x["target"])
                            t = x["target"]
                            break
            for j in range(0,len(edges)):
                for i,x in enumerate(edges):
                        if re.search("OutImage",x["target"]):
                            nodes_with_out[x["source"]]=x["target"]
            print("############# ",que)
            for idx,i in enumerate(que):
                if i in nodes_with_out:
                    inps[i]["output_node"] = nodes_with_out[i]
                else:
                    inps[i]["output_node"] = ""
                if idx > 0:
                    inps[i]["prev_node"] = {}
                    for k in outs[que[idx-1]]:
                        
                        if k != "prev_node":
                            inps[i]["prev_node"][k] = outs[que[idx-1]][k]
                        else:
                            for pk in outs[que[idx-1]]["prev_node"]:
                                if pk not in inps[i]["prev_node"]:
                                    inps[i]["prev_node"][pk] = outs[que[idx-1]]["prev_node"][pk]
                    
                    for pk in inps[que[idx-1]]["prev_node"]:
                        if pk not in inps[i]["prev_node"]:
                            inps[i]["prev_node"][pk] = inps[que[idx-1]]["prev_node"][pk]
                else:
                    inps[i]["prev_node"] = ""
                if str(i.split("_")[0]) == "CustomNode":
                    writelog(f'{str(inps[i]["custom_node"]).replace("_"," ")} Node in Progress...')
                elif str(i.split("_")[0]) == "End":
                    writelog(f'Done Executing')
                else:
                    writelog(f'{str(i.split("_")[0])} Node in Progress...')
                outs[i] = functs[str(i.split("_")[0])](inps[i])
        else:
            outs = {}
            que = []
            nodes_with_out = {}
            t = "Loop"
            for j in range(0,len(edges)):
                for i,x in enumerate(edges):
                    if re.search(t,edges[str(x)]["source"]):
                        if not re.search("OutImage",edges[str(x)]["target"]):
                            que.append(edges[str(x)]["target"])
                            t = edges[str(x)]["target"]
                            break
            for j in range(0,len(edges)):
                for i,x in enumerate(edges):
                        if re.search("OutImage",edges[str(x)]["target"]):
                            nodes_with_out[edges[str(x)]["source"]]=edges[str(x)]["target"]
            for idx,i in enumerate(que):
                if i in nodes_with_out:
                    inps[i]["output_node"] = nodes_with_out[i]
                else:
                    inps[i]["output_node"] = ""
                if idx > 0:
                    inps[i]["prev_node"] = {}
                    for k in outs[que[idx-1]]:
                        if k != "prev_node":
                            inps[i]["prev_node"][k] = outs[que[idx-1]][k]
                        else:
                            for pk in outs[que[idx-1]]["prev_node"]:
                                if pk not in inps[i]["prev_node"]:
                                    inps[i]["prev_node"][pk] = outs[que[idx-1]]["prev_node"][pk]
                    for pk in inps[que[idx-1]]["prev_node"]:
                        if pk not in inps[i]["prev_node"]:
                            inps[i]["prev_node"][pk] = inps[que[idx-1]]["prev_node"][pk]
                else:
                    try:
                        inps[i]["prev_node"] = loopout[que[-1]]
                    except:
                        inps[i]["prev_node"] = ""
                if str(i.split("_")[0]) == "CustomNode":
                    writelog(f'{str(inps[i]["custom_node"]).replace("_"," ")} Node in Progress...')
                elif str(i.split("_")[0]) == "End":
                    writelog(f'Done Executing')
                else:
                    writelog(f'{str(i.split("_")[0])} Node in Progress...')
                outs[i] = functs[str(i.split("_")[0])](inps[i])
        return outs
    except Exception as e: 
        print(e)
        writelog("An error happened. Please check your terminal/console for more details, and restart the process")
        return {"null":"null"}











