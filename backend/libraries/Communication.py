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

from paho.mqtt import client as mqtt_client

for key, value in list(globals().items()):
    if callable(value):
        if "function" in str(type(value)):
            importedfuncs.append(key)
        
newmethods={'MQTT_Publish': {'0': ['text', 'Client ID'], '1': ['text', 'Username'], '2': ['text', 'Password'], '3': ['text', 'Broker'], '4': ['text', 'Port'],'5': ['text', 'Message'], '6': ['text', 'Topic']},'MQTT_Subscribe': {'0': ['text', 'Client ID'], '1': ['text', 'Username'], '2': ['text', 'Password'], '3': ['text', 'Broker'], '4': ['text', 'Port'],'5': ['text', 'Topic']}}
color={'MQTT_Publish': '#1b8019','MQTT_Subscribe': '#1b8019'}


def MQTT_Publish(inps):
    client = mqtt_client.Client(inps["vars"]["Client ID"] if "Client ID" in inps["vars"] else "")
    if "Username" in inps["vars"] and inps["vars"]["Username"] != "":
        client.username_pw_set(inps["vars"]["Username"], inps["vars"]["Password"])
    client.connect(inps["vars"]["Broker"], int(inps["vars"]["Port"]))
    client.loop_start()
    msg = inps["prev_node"][inps["vars"]["Message"]] if inps["vars"]["Message"] in inps["prev_node"] else inps["vars"]["Message"]
    result = client.publish(inps["vars"]["Topic"], msg)
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{inps['vars']['Topic']}`")
    else:
        print(f"Failed to send message to topic {inps['vars']['Topic']}")
    client.loop_stop()
    return {"message":msg,"outimagenode":inps["output_node"]}

def MQTT_Subscribe(inps):
    client = mqtt_client.Client(inps["vars"]["Client ID"] if "Client ID" in inps["vars"] else "")
    def on_message(client, userdata, msg):
        writelog(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
    if "Username" in inps["vars"] and inps["vars"]["Username"] != "":
        client.username_pw_set(inps["vars"]["Username"], inps["vars"]["Password"])
    client.connect(inps["vars"]["Broker"], int(inps["vars"]["Port"]))
    client.loop_start()
    client.subscribe(inps["vars"]["Topic"])
    client.on_message = on_message
    client.loop_forever()
    return {"outimagenode":inps["output_node"]}
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
