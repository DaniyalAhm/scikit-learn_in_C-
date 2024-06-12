# Use an official Ubuntu base image
FROM ubuntu:latest

# Install g++, make, and other C++ build essentials
RUN apt-get update && \
    apt-get install -y g++ make cmake libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /home

# Copy your source code
COPY . /home

