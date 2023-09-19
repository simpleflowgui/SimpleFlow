import subprocess,webbrowser,os

webbrowser.open("http://localhost:5173/")
subprocess.run(["npm", "run","dev"], cwd=f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/my-react-flow-app")

