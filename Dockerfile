FROM python:3.9

# Set environment variables for configuration and defaults
ENV TEST_DIR=/app/tests
ENV MESHE_DIR=/app/meshe
ENV FVM_DIR=/app/fvm
ENV TSTEP_DIR=/app/tstep
ENV SOLVER_DIR=/app/solver
ENV FEM_DIR=/app/fem
ENV TEST_FILE=test_layers.py.py
# Set the working directory inside the container
WORKDIR /app

# Copy the test files and sources into the container working directory
COPY tests $TEST_DIR
COPY meshe $MESHE_DIR
COPY fvm $FVM_DIR
COPY tstep $TSTEP_DIR
COPY solver $SOLVER_DIR
COPY fem $FEM_DIR
COPY requirements.txt /app/
COPY __init__.py /app/__init__.py

RUN pip install --no-cache-dir -r requirements.txt

CMD pytest tests
#CMD ["python", "tutorials/T1_curvefitting.py"]