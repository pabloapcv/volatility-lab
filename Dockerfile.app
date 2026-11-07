FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including Node.js and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    supervisor \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy Python project files (hatchling needs README/LICENSE to build)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY backend/ backend/
COPY data/ data/
COPY scripts/ scripts/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy and build frontend
COPY frontend/ frontend/
WORKDIR /app/frontend
RUN npm ci && npm run build
WORKDIR /app

# Create supervisor config
RUN mkdir -p /var/log/supervisor
COPY <<EOF /etc/supervisor/conf.d/app.conf
[supervisord]
nodaemon=true
user=root

[program:backend]
command=uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
directory=/app
environment=PYTHONPATH="/app/src:/app"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:frontend]
command=npm run dev -- --host 0.0.0.0 --port 5175
directory=/app/frontend
environment=VITE_API_URL="",VITE_PROXY_TARGET="http://127.0.0.1:8000"
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
EOF

# Set environment
ENV PYTHONPATH=/app/src:/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000 5175

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
