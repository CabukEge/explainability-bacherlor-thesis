# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Copy all project files into the container
COPY . /app

# Make entrypoint script executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create artifacts directory
RUN mkdir -p /app/artifacts

# Upgrade pip, install dependencies, then install the project in editable mode
RUN pip install --upgrade pip && \
    pip install numpy==1.23.5 && \
    pip install matplotlib==3.7.1 seaborn==0.12.2 pandas==2.0.0 && \
    pip install -e .

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command runs all tests (and thus training, evaluation, and matrix generation)
CMD ["python", "run_tests.py", "--approach", "all"]