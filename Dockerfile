# ==========================================
# STAGE 1: The Builder
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Create a virtual environment so we can easily copy all dependencies later
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in ARGs for MLflow
ARG RUN_ID
ARG DAGSHUB_TOKEN

# Set ENV vars for the download step
ENV MLFLOW_TRACKING_URI="https://dagshub.com/17-doha/GANS.mlflow"
ENV MLFLOW_TRACKING_USERNAME="17-doha"
ENV MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN

# Copy the rest of the code
COPY . .

# Download the model and generate data in the builder stage
RUN mlflow artifacts download --run-id $RUN_ID --dst-path ./downloaded_model
RUN python -m app.data_generator

# ==========================================
# STAGE 2: The Production Runner
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy ONLY the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the application code, the downloaded model, and generated data
COPY --from=builder /app /app

# Set the startup command
CMD ["python", "-m", "app.main"]