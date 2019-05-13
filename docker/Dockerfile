FROM continuumio/miniconda3

MAINTAINER Gilles Bodart <gillesbodart@users.noreply.github.com>

RUN conda create -n env python=3.6
RUN echo "source activate env" > ~/.bashrc

ENV PATH /opt/conda/envs/env/bin:$PATH
 
RUN apt-get -qq -y update
RUN apt-get -qq -y upgrade
RUN apt-get -qq -y install \
        gcc \
        g++ \
        wget \
        curl \
        git \
        make \
        sudo \
        bash-completion \
        tree \
        vim

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

<<<<<<< HEAD:docker/Dockerfile
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH
=======
# Install any needed packages specified in requirements.txt
WORKDIR /app
COPY ./docker/requirements.txt /app/requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt --verbose
>>>>>>> e5e8b398b7b03a849dea3dc9355c7c6a66268113:DOCKERFILE

# Download LASER from FB and install
RUN git clone https://github.com/facebookresearch/LASER.git
ENV LASER /app/LASER
WORKDIR $LASER  
RUN bash ./install_models.sh
RUN bash ./install_external_tools.sh
COPY ./docker/decode.py $LASER/tasks/embed/decode.py

<<<<<<< HEAD:docker/Dockerfile
# Make port 80 available to the world outside this container
=======
# run tests
>>>>>>> e5e8b398b7b03a849dea3dc9355c7c6a66268113:DOCKERFILE
WORKDIR /app
RUN echo "Hello World" > test.txt
RUN $LASER/tasks/embed/embed.sh test.txt en test_embed.raw
RUN python $LASER/tasks/embed/decode.py test_embed.raw

# copy only at the end so that code updates do not require new slow build
COPY ./docker /app  

EXPOSE 8099 
CMD ["python", "app.py"] 
