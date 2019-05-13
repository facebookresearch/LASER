FROM python:3.7-stretch

MAINTAINER Gilles Bodart <gillesbodart@users.noreply.github.com>
 
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

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./docker /app


# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt --verbose


# Download LASER from FB
RUN git clone https://github.com/facebookresearch/LASER.git

ENV LASER /app/LASER
WORKDIR $LASER  

RUN bash ./install_models.sh
RUN bash ./install_external_tools.sh

COPY ./docker/decode.py $LASER/tasks/embed/decode.py


# Make port 80 available to the world outside this container
WORKDIR /app

RUN echo "Hello World" > test.txt

RUN $LASER/tasks/embed/embed.sh test.txt en test_embed.raw
RUN python $LASER/tasks/embed/decode.py test_embed.raw

#Open the port 80
EXPOSE 80 
 
CMD ["/bin/bash"] 