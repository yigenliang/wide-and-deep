# Use an official Tensorflow runtime as a base image, Ubuntu xenial source
FROM tensorflow/tensorflow:1.1.0-py3

# Install ping for testing
#RUN apt-get update && apt-get install -y inetutils-ping lsof wget dnsutils

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Setup ENTRYPOINT
ENTRYPOINT ["python", "census_wide_n_deep.py"]
