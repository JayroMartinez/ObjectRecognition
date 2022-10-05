# DOCKERFILE

FROM python:3.8-slim-bullseye

# COPY /PyCode . 

RUN pip3 install numpy scikit-learn pandas jupyterlab matplotlib seaborn pingouin

EXPOSE 8888

CMD ["/bin/bash", "-c", "jupyter lab --ip='0.0.0.0' --no-browser --allow-root"]

