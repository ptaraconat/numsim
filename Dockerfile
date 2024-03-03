FROM python:3.9

# Set environment variables for configuration and defaults
ENV TEST_DIR=/app/tests
ENV MESHE_DIR=/app/meshe
ENV FVM_DIR=/app/fvm
ENV TEST_FILE=test_layers.py.py
# Set the working directory inside the container
WORKDIR /app

# Copy the test files and sources into the container working directory
COPY tests $TEST_DIR
COPY meshe $MESHES_DIR
COPY fvm $FVM_DIR
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD pytest tests
#CMD ["python", "tutorials/T1_curvefitting.py"]