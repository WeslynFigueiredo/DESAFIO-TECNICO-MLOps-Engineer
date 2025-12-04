.PHONY: help venv install data train infer api test docker-build docker-run streamlit

help:
	@echo "Comandos disponíveis:"
	@echo "  make install       - instalar dependências"
	@echo "  make data          - preparar dados (data_prep)"
	@echo "  make train         - treinar modelo (MLflow + joblib)"
	@echo "  make infer         - rodar inferência via script"
	@echo "  make api           - subir API FastAPI local"
	@echo "  make test          - rodar pytest"
	@echo "  make docker-build  - build da imagem Docker"
	@echo "  make docker-run    - rodar container Docker"
	@echo "  make streamlit     - rodar app Streamlit"

install:
	pip install -r requirements.txt

data:
	python -m src.data_prep

train:
	python -m src.train

infer:
	python -m src.infer

api:
	uvicorn src.api.main:app --reload

test:
	pytest

docker-build:
	docker build -t fish-weight-api .

docker-run:
	docker run -p 8000:8000 fish-weight-api

streamlit:
	streamlit run app_streamlit.py
