FROM continuumio/miniconda3

MAINTAINER Gilles Bodart <gillesbodart@users.noreply.github.com>

# Install build-essential (compiler and development tools)
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN conda create -n env python=3.8
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# Set the working directory to /app
WORKDIR /app

# Copy the local laser-encoders repository
COPY laser_encoders /app/laser_encoders
COPY pyproject.toml /app/pyproject.toml

RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install Flask==2.3.3 Requests==2.31.0

# Define the argument for language
ARG langs="eng_Latn"

# Download language models for each specified language
RUN for lang in $langs; do \
        python -m laser_encoders.download_models --lang=$lang; \
    done

# Open the port 80
EXPOSE 80

COPY docker/app.py /app/app.py

CMD ["/bin/bash"]
