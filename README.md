# SimpleFlow
A simple to use node-based GUI for creating multipurpose flows, such as those for training models to perform computer vision tasks.

<a href="alternative text"><img src="https://github.com/simpleflowgui/SimpleFlow/blob/main/simpleflow.png" align="middle" width="800" height="400"></a>

# Installation
## 1. Install Node.js and npm

(Linux)
```code
sudo apt update
sudo apt install Node.js
sudo apt install npm
```

(Windows)

Download and install https://nodejs.org/en/download

## 2. Install Python and pip

(Linux)
```code
sudo apt update
sudo apt-get install python3
sudo apt-get install python3-pip python-dev
```

(Windows)

Download and install https://www.python.org/downloads/windows/

## 3. Install Git
(Linux)
```code
sudo apt-get install git
```

(Windows)
Download and install https://git-scm.com/download/win

## 4. Clone SimpleFlow github repository
```code
git clone https://github.com/simpleflowgui/SimpleFlow
```

## 5. Create a new React project using Vite
```code
cd SimpleFlow
npm create vite@latest my-react-flow-app -- --template react
npm install reactflow --force
cd my-react-flow-app
npm install
```

## 6. Run the setup file

(Linux)
```code
cd SimpleFlow
sudo chmod +x setup.sh
./setup.sh
```

(Windows)
```code
cd SimpleFlow
setup.cmd
install.cmd
```

## 7. Run SimpleFlow
(Linux)
```code
simpleflow
```

(Windows)
```code
simpleflow.cmd
```




