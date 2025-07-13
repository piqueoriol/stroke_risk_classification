# Image name
IMAGE_NAME := stroke-predictor

# Default data directory
DATA_DIR := $(PWD)/data

# Build the Docker image
docker-build:
	docker build -t $(IMAGE_NAME) .

# Run the container, binding your host data/ folder to /app/data in the container
docker-run: docker-build
	docker run --rm \
	  -v "$(DATA_DIR):/app/data" \
	  $(IMAGE_NAME)