FROM --platform=linux/amd64 python:3.9.11-bullseye

RUN apt install curl
RUN apt-get update


# devtools is required for installing prophet
RUN apt-get install -y gcc  \
    g++ \
    build-essential  \
    python-dev \
    python3-dev

# install pyodbc
RUN pip3 install --upgrade pip

# Copy the requirements file and install dependencies

# Create the app directory and set it as the working directory
WORKDIR /app
# Expose the port the application will run on
EXPOSE 8080

RUN pip install --upgrade cython numpy==1.26.2

COPY requirements.txt .
RUN pip install -r --no-cache-dir requirements.txt

# Copy the rest of the application code, this is done last to avoid running requirements.txt on every code change
COPY . .
# Set the default command to run the application with Gunicorn
CMD ["gunicorn", "server", "--model", "models/best_ICELSTMAmodel_ood.pth", "--port", "8080"]