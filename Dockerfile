# DOCKERFILE

FROM python:3.8-slim-bullseye

ADD /PyCode . 

RUN pip3 install numpy scikit-learn pandas jupyterlab

# CMD ["python3", "./ObjectClassification_kin.py"]
CMD ["python3"]