FROM python:3.13-slim-trixie
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python run_pipeline.py
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser -m -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 5000
ENV FLASK_APP=app/api.py
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
