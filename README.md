# AWS-docker-container

In this project, I developed a sklearn app with flask on AWS cloud9, and then I containerized it and pushed it to dockerhub for better scalibility.

## Create a `requirements.txt`
```bash
Flask==1.0.2
pandas==0.24.2
scikit-learn==0.20.3
```

## Create a `Makefile`
```bash

setup:
	python3 -m venv ~/.docker

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	#python -m pytest -vv --cov=myrepolib tests/*.py
	#python -m pytest --nbval notebook.ipynb


lint:
	hadolint Dockerfile 
	pylint --disable=R,C,W1203 app.py

all: install lint test
```

## Create a virtualenv
```
python3 -m venv ~/.docker
# Activate the virtualenv
source ./.docker/bin/activate
# Install all the requirements
make all
```

## Write your Docker file
```bash
#From Image selected
FROM python:3.7.3-stretch

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . app.py /app/

# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --upgrade pip &&\
    pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 80
EXPOSE 8080

# Run app.py at container launch
CMD ["python", "app.py"]
```

## Write your app `main.py`
*Caveat! you need to have a `main.py` script in order to deploy an app. If not GSC cannot interact with your app.*

```python
from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

def scale(payload):
    """Scales Payload"""

    LOG.info(f"Scaling Payload: {payload}")
    scaler = StandardScaler().fit(payload)
    scaled_adhoc_predict = scaler.transform(payload)
    return scaled_adhoc_predict

@app.route("/")
def home():
    html = f"<h3>Sklearn Prediction Home</h3>"
    return html.format(format)

@app.route("/predict", methods=['POST'])
def predict():
    json_payload = request.json
    LOG.info(f"JSON payload: {json_payload}")
    inference_payload = pd.DataFrame(json_payload)
    LOG.info(f"inference payload DataFrame: {inference_payload}")
    scaled_payload = scale(inference_payload)
    prediction = list(clf.predict(scaled_payload))
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    clf = joblib.load("boston_housing_prediction.joblib")
    app.run(host='0.0.0.0', port=8080, debug=True)
```
## Build and deploy your image to Docker Hub
```bash
#Select the name of your image
docker build --tag=flasksklearn .

# Set a dockerpath with user and name of the image
dockerpath="mokszekei/flasksklearn"

# Authenticate & Tag
echo "Docker ID and Image: $dockerpath"
docker login && docker image tag describer $dockerpath

# Push Image
docker image push $dockerpath 

#To run the container. You would need to expose a port to connect with the docker port.
#In this case I am using 8080 for both.
docker run -p 8080:8080 -it mokszekei/flasksklearn bash   

#run the flask app
python3 main.py
```
