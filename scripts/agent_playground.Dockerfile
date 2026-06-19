FROM python:3.9-slim-bullseye
RUN apt-get update && apt-get install -y --no-install-recommends git bash && rm -rf /var/lib/apt/lists/*
WORKDIR /app