# Use a lightweight Python runtime
FROM python:3.9-slim

# Create app directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Declare the data folder as a volume
VOLUME ["/app/data"]

# Default command: run main.py with the config
ENTRYPOINT ["python", "src/main.py", "--config", "config.yaml"]