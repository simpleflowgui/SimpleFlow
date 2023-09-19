pip install virtualenv
cd backend
python -m venv cv
cv\Scripts\activate
pip install -r requirements.txt
cd ..
cd my-react-flow-app
rm -r src
cd ..
mv src my-react-flow-app/src
