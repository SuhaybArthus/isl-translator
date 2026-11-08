FROM python:3.11-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY api ./api
COPY web ./web
COPY src/inference ./src/inference
COPY models/full_tf_s1.onnx models/full_tf_s1.meta.json ./models/
ENV ISL_MODEL=full_tf_s1
ENV ISL_RECORDING=0
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]