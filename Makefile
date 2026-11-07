API_PORT ?= 8765
FRONTEND_PORT ?= 5175
API_HOST ?= 0.0.0.0

.PHONY: install demo test docker-demo

install:
	pip install -e ".[dev,ml]"
	cd frontend && npm ci

demo: install
	API_PORT=$(API_PORT) FRONTEND_PORT=$(FRONTEND_PORT) API_HOST=$(API_HOST) bash scripts/demo.sh

test:
	PYTHONPATH=src:. pytest tests/ -v

docker-demo:
	docker compose up --build
