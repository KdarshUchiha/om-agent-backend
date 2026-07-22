FROM python:3.11-slim

WORKDIR /app

# Node.js provides `node --check` for full JavaScript syntax validation in the
# verifier. Without it the verifier falls back to a brace-balance heuristic.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
