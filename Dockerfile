# API + website. LLM = Groq (set GROQ_API_KEY) or any OpenAI-style server (set LLM_URL).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag.py app.py ./
COPY static ./static
COPY data ./data
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
