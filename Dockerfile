# DOCKERFILE

FROM python:3.8-slim-bullseye

ADD /PyCode . 

RUN pip3 install numpy scikit-learn pandas jupyterlab

EXPOSE 8888

CMD ["/bin/bash", "-c", "jupyter lab --ip='0.0.0.0' --no-browser --allow-root"]

