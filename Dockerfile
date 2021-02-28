# Base image
FROM python:3.8

# Create an 'app' folder and this becomes the initial working directory
WORKDIR /app

# Add all files from current directory of PC (where Dockerfile is stored) to the working directory of docker image
ADD . .

# Install packages
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Run the main code file
CMD ["python", "main.py"]
