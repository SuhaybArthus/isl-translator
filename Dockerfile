FROM python:3.11-slim
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY api ./api
COPY web ./web
COPY src/inference ./src/inference
COPY models/final_v1.onnx models/final_v1.meta.json ./models/

RUN useradd --create-home app && chown -R app /app
USER app

ENV ISL_MODEL=final_v1 ISL_RECORDING=0 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "16384"]