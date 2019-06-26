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
        unzip \
        sudo \
        vim

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Set the working directory to /app
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt --verbose


# Download LASER from FB
RUN git clone https://github.com/facebookresearch/LASER.git

ENV LASER /app/LASER
WORKDIR $LASER

RUN bash ./install_models.sh


#Installing FAISS

RUN conda install --name env -c pytorch faiss-cpu -y

RUN bash ./install_external_tools.sh

COPY ./decode.py $LASER/tasks/embed/decode.py


# Make port 80 available to the world outside this container
WORKDIR /app

RUN echo "Hello World" > test.txt

RUN $LASER/tasks/embed/embed.sh test.txt en test_embed.raw
RUN python $LASER/tasks/embed/decode.py test_embed.raw

#Open the port 80
EXPOSE 80

COPY ./app.py /app/app.py

CMD ["/bin/bash"]
