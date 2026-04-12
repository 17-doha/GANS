FROM python:3.10-slim
WORKDIR /app

ARG RUN_ID
ARG DAGSHUB_TOKEN

ENV MLFLOW_TRACKING_URI="https://dagshub.com/17-doha/GANS.mlflow"
ENV MLFLOW_TRACKING_USERNAME="17-doha"
ENV MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

RUN mlflow artifacts download --run-id $RUN_ID --dst-path ./downloaded_model

RUN python -m app.data_generator

CMD ["python", "-m", "app.main"]
