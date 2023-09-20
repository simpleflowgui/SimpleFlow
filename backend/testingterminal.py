import subprocess,webbrowser,os

webbrowser.open("http://localhost:5173/")
try:
  subprocess.run(["npm", "run","dev"], cwd=f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/my-react-flow-app")
except:
  subprocess.run(["npm", "run","dev"], cwd=f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}\my-react-flow-app",shell=True)

