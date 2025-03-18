# Use an Ubuntu base image
FROM ubuntu:20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libsdl2-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libpng-dev \
    libjpeg-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    zlib1g-dev \
    timidity \
    tar \
    nasm \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install ViZDoom
RUN pip3 install vizdoom
RUN pip install pygame
RUN pip install opencv-python

COPY ViZDoom-master/scenarios/level1.cfg /app/scenarios/

# Set the working directory
WORKDIR /app

# Copy your Python script and WAD file
COPY /ViZDoom-master/examples/python/level1.py .
COPY MAP01.wad .
COPY ViZDoom-master/scenarios/MAP01.wad /app/scenarios/

# Run the script
CMD ["python3", "level1.py"]