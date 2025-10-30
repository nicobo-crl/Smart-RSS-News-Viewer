# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache which reduces image size
# --trusted-host pypi.python.org: Can help in some network environments
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application's code into the container
# This includes app.py and the templates directory
COPY . .

# Expose the port the app runs on
EXPOSE 4000

# Define environment variables (optional, but good practice)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=4000

# Command to run the application
# Use gunicorn for a production-ready server or flask run for development
# Using "flask run" as per the original script's __main__ block
CMD ["python", "app.py"]```

### Project Structure

Your final project directory should look like this: