# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Copy all project files into the container
COPY . /app

# Upgrade pip, install a compatible version of numpy, then install the project in editable mode.
RUN pip install --upgrade pip && \
    pip install numpy==1.23.5 && \
    pip install -e .

# Default command runs all tests (and thus training, evaluation, and matrix generation)
CMD ["python", "run_tests.py", "--approach", "all"]
