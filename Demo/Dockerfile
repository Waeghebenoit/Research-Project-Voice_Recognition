# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    python3-dev \
    libffi-dev \
    libatlas-base-dev \
    portaudio19-dev \
    && apt-get clean

# Set the working directory in the container
WORKDIR /apps/multiword-model

# Copy requirements.txt into the container
COPY requirements.txt .

# Copy the entire project directory (including models) into the container
COPY . /apps/multiword-model

# Copy the models directory explicitly into the container
COPY models /apps/models

# Install any needed Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Define environment variable
ENV NAME MultiwordModel

# Run app.py when the container launches
# Change the app to the script you want to run
CMD ["python", "path/to/your/script.py"]
