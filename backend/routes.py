from flask import current_app,jsonify,request, Flask
from app import create_app
import webbrowser
import Nodes
import os,json, pyautogui, time
try:
	from signal import SIGKILL
except:
	from signal import SIGABRT
from multiprocessing import Process, Queue

def my_function(q,edges,inps,loopout,lf):
	res = Nodes.mainfunc(edges,inps,loopout,lf)
	json_object = json.dumps(res, indent=4)
	with open("./nodeoutput.json","w") as file:
		file.write(json_object)
	return 0

# Create an application instance
app = create_app()

@app.route("/")
def npmserver():
	time.sleep(0.1)
	pyautogui.hotkey('ctrl', 'w')
	with open("testingterminal.py") as f:
		exec(f.read())


@app.route("/saveflow",methods=['GET', 'POST','DELETE'])
def saveflow():
	flow = request.json["flow"]
	filename = request.json["filename"]
	return Nodes.saveflow(flow,filename)



@app.route("/upload",methods=['GET', 'POST','DELETE'])
def uploadfile():
	print("trigerred")
	node = request.json["node"]
	inps = request.json['inps']
	return Nodes.upload(node,inps)

@app.route("/stopprocess",methods=['GET', 'POST','DELETE'])
def stopprocess():
	with open("./processpid.txt","r") as file:
		pid = file.read()
	try:
		os.kill(int(pid), SIGKILL)
	except:
		os.kill(int(pid), SIGABRT)
	return "none"


@app.route("/communicate",methods=['GET', 'POST','DELETE'])
def test():
	if request.method == 'POST':
		edges = request.json['edges']
		inps = request.json['inps']
		loopout = request.json['loopout']
		lf = request.json['lf']
		queue = Queue()
		p = Process(target=my_function, args=(queue,edges,inps,loopout,lf))
		p.start()
		p.join() 
		with open("./nodeoutput.json","r") as file:
			out = json.load(file)   
		return out
	

@app.route("/clearlog",methods=['GET', 'POST','DELETE'])
def clearlog():
	if request.method == 'POST':
		return Nodes.clearlog()


@app.route("/flows",methods=['GET', 'POST','DELETE'])
def getflowslist():
	return Nodes.getflowslist()


@app.route("/loadnodes",methods=['GET', 'POST','DELETE'])
def loadnodes():
	return Nodes.loadnodes()


@app.route("/savelibrary",methods=['GET', 'POST','DELETE'])
def savelibrary():
	lib = request.json['library']
	imports = request.json['imports']
	text = request.json['text']
	inputmethods = request.json['methods']
	colors = request.json['colors']
	return Nodes.savelibrary(lib,imports,text,inputmethods,colors)


@app.route("/loadcams",methods=['GET', 'POST','DELETE'])
def loadcams():
	return Nodes.loadcams()


if __name__ == "__main__":
	webbrowser.open("http://localhost:5000")
	app.run(debug=False)
