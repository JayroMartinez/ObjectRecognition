# DOCKERFILE

FROM python:3.8-slim-bullseye

# COPY /PyCode . 

RUN pip3 install numpy scikit-learn pandas jupyterlab matplotlib seaborn pingouin scipy statannotations statannot torch==1.5.1+cpu torchvision==0.6.1+cpu

EXPOSE 8888

CMD ["/bin/bash", "-c", "jupyter lab --ip='0.0.0.0' --no-browser --allow-root"]

