# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the necessary files into the container
COPY ./dist/sonocop /usr/src/app
COPY ./dist/sonocop/_internal /usr/src/app/_internal

RUN apt-get update && apt-get install -y libffi-dev

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/src/app/_internal

# Define environment variable
ENV NAME sonocop

# Run sonocop when the container launches
ENTRYPOINT ["./sonocop", "/data"]
