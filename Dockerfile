FROM python:3.9-slim

# Copy local code to the container image.
ADD . /app
WORKDIR /app

# Install production dependencies.
RUN pip3 install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Deploy app using gunicorn
CMD exec gunicorn wsgi:server --bind :$PORT