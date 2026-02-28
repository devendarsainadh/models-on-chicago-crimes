FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir pyspark==3.5.1 pandas==2.2.2 scikit-learn==1.5.1 pyarrow==17.0.0 pyyaml==6.0.2 pytest==8.3.3

CMD ["python", "scripts/run_pipeline.py"]
