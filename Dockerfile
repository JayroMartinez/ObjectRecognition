# DOCKERFILE

# FROM python:3.8-slim-bullseye
FROM python:3.8

# COPY /PyCode . 

RUN apt-get update
RUN apt-get install git
RUN pip3 install --upgrade pip
RUN pip3 install numpy scikit-learn pandas jupyterlab matplotlib seaborn pingouin scipy statannotations statannot torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html trimesh git+https://github.com/mattloper/chumpy git+https://github.com/otaheri/MANO

EXPOSE 8888

# CMD ["/bin/bash", '-c", "cp -r /PyCode/mano/ /usr/local/lib/python3.8/site-packages"]
CMD ["/bin/bash", "-c", "jupyter lab --ip='0.0.0.0' --no-browser --allow-root"]
