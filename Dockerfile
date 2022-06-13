# DOCKERFILE

FROM python:3.8-slim-bullseye

ADD . . 

RUN pip install json numpy scikit-learn pandas

CMD ["python", "./PyCode/ObjectClassification_kin.py"]