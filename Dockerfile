FROM python:3.13-slim-trixie as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./data .
COPY ./features .
COPY ./models .
COPY ./requirements.txt .
COPY ./run_pipeline.py .
RUN python run_pipeline.py

FROM python:3.13-slim-trixie as final
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir ./features
RUN mkdir ./models
RUN mkdir ./data
COPY ./data ./data/fetch_price.py
COPY ./features/technical_indicators.py ./features/
COPY ./requirements.txt .
COPY --from=builder /app/models/saved ./models/saved
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser -m -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 5000
ENV FLASK_APP=app/api.py
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
