FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.lock ./
COPY vendor/langchain-google-firestore/dist/langchain_google_firestore-0.5.0.post2-py3-none-any.whl ./vendor/langchain-google-firestore/dist/
COPY pyproject.toml ./
COPY app ./app
COPY model ./model
RUN echo "9ced9b491908b453a31a4b2598c1ec5fbbddf106dff868adcfe36bd5e60124c2  vendor/langchain-google-firestore/dist/langchain_google_firestore-0.5.0.post2-py3-none-any.whl" | sha256sum -c - \
    && pip install --no-cache-dir --require-hashes -r requirements.lock \
    && pip install --no-cache-dir --no-deps . \
    && pip check \
    && sha256sum requirements.lock > requirements.lock.sha256

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-graceful-shutdown", "25"]
