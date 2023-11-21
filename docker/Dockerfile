FROM continuumio/miniconda3

MAINTAINER Gilles Bodart <gillesbodart@users.noreply.github.com>

# Install build-essential (compiler and development tools)
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN conda create -n env python=3.9
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# Set the working directory to /app
WORKDIR /app

# Copy the local laser-encoders repository
COPY ../laser_encoders /app/laser_encoders

RUN pip3 install --upgrade pip
RUN pip install /app/laser_encoders --verbose

#Open the port 80
EXPOSE 80

COPY ./app.py /app/app.py

CMD ["/bin/bash"]
