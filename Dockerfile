# Using this image to utilize GPU, cuda, and cudnn
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Use this image for CPU loads to get faster build times 
FROM debian:bookworm-slim

# Install uv python package manager
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy recipe files in the docker image
ENV RECIPE_DIR=/opt/recipe
WORKDIR $RECIPE_DIR

# Copy only dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies (cached if deps didn't change)
RUN uv sync --no-cache --locked

# Now copy the rest of the recipe (source) files
COPY . .

# Adds the virtual environment in path to make it "system wide"
# Commands can be executed with out 'uv run': "uv run hafnia profile ls" -> "hafnia profile ls"
ENV PATH="${RECIPE_DIR}/.venv/bin:$PATH"
