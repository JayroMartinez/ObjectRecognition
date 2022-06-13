# DOCKERFILE

FROM python:3.8-slim-bullseye

ADD /PyCode . 

RUN pip install numpy scikit-learn pandas

CMD ["python", "./ObjectClassification_kin.py"]