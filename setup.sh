chmod +x simpleflow
export PATH=$PATH:/home/mohammad/SimpleFlow
sudo cp simpleflow /usr/bin/
pip install virtualenv
cd backend
virtualenv cv
source cv/bin/activate
pip install -r requirements.txt
sudo apt-get install python-tk
cd ..
cd my-react-flow-app
rm -r src
cd ..
mv src my-react-flow-app/src

