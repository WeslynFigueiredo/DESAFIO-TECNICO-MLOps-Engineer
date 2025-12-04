# Fish Weight MLOps

Projeto desenvolvido para o desafio técnico de **MLOps Engineer (Python + ML + Container/Lambda)**.  
Objetivo: prever o **peso de peixes** a partir de medidas físicas e também a partir de **imagens**, calculando inclusive a **biomassa do tanque**, com boas práticas de MLOps, API em container, dashboard e CI.

## Arquitetura da solução

Fluxo principal:

1. `src/data_prep.py`  
   - Lê o dataset Fish Market.  
   - Limpa/organiza colunas.  
   - Faz split em `train.csv` e `test.csv` (em `data/processed/`).

2. `src/train.py`  
   - Lê `train.csv`.  
   - Separa features (`Length1`, `Length2`, `Length3`, `Height`, `Width`) e target (`Weight`).  
   - Faz split treino/validação.  
   - Treina um modelo `LinearRegression` (Scikit-Learn).  
   - Calcula **MAE**.  
   - Salva o modelo em `models/linear_regression_fish.joblib`.  
   - Loga **parâmetros, métrica e modelo** no **MLflow** (experimento `fish_weight_regression`).

3. `src/infer.py`  
   - Carrega o modelo salvo.  
   - Expõe a função `predict_weight(...)` usada por:
     - script de inferência,
     - API FastAPI,
     - app Streamlit.

4. API (`src/api/main.py`)  
   - `GET /`: healthcheck.  
   - `POST /predict`: recebe medidas numéricas em JSON e devolve `predicted_weight`.  
   - `POST /predict-image`: recebe uma **imagem do peixe** (`multipart/form-data`) e um parâmetro `quantity` (quantidade de peixes no tanque).  
     - Extrai largura e altura em pixels da imagem.  
     - Deriva um conjunto de medidas compatíveis com o modelo tabular (Length1, Length2, Length3, Height, Width) a partir dessas dimensões, simulando um módulo de visão computacional.  
     - Chama internamente `predict_weight` para estimar o peso individual.  
     - Calcula a **biomassa estimada do tanque** em kg: `biomass_kg = predicted_weight * quantity / 1000`.  
     - Resposta inclui:
       - `image_width_px`, `image_height_px`  
       - `features_used` (medidas derivadas)  
       - `predicted_weight` (peso individual estimado)  
       - `quantity`  
       - `biomass_kg` (biomassa total estimada do tanque).  

   Esse endpoint simula o fluxo usado em sistemas reais com câmeras subaquáticas: **frame de vídeo → extração de medidas → peso individual → biomassa do tanque**.

5. Container (Docker)  
   - Dockerfile expõe a API FastAPI na porta 8000.

6. Streamlit (`app_streamlit.py`)  
   - Interface web para interação com o modelo:
     - **Aba “Medidas manuais”**: formulário para o usuário informar Length1, Length2, Length3, Height e Width. Chama `POST /predict` e exibe o peso previsto.  
     - **Aba “Imagem do peixe”**: permite upload de uma foto de peixe e informar a **quantidade de peixes no tanque**.  
       - Exibe a imagem.  
       - Chama `POST /predict-image` com a imagem e `quantity`.  
       - Mostra:
         - largura/altura da imagem em pixels,  
         - as medidas derivadas usadas pelo modelo,  
         - o peso previsto a partir da imagem,  
         - a **biomassa estimada do tanque em kg**.

7. Testes (`tests/test_infer.py`, `tests/test_api.py`)  
   - Testes unitários garantem que `predict_weight` retorna valores numéricos válidos em diferentes faixas.  
   - Teste de integração da API verifica `POST /predict` usando `TestClient` do FastAPI.

## Como executar

### Pré‑requisitos

- Python 3.10+  
- Docker (opcional, para rodar em container)

### Instalar dependências

pip install -r requirements.txt

ou

make install


### Preparar dados

make data


Gera `data/processed/train.csv` e `test.csv`.

### Treinar o modelo (com MLflow)

make train


- Treina o modelo.  
- Imprime o **MAE**.  
- Loga o experimento no MLflow (`mlruns/`).

Opcional – abrir MLflow UI:

python -m mlflow ui

acessar http://127.0.0.1:5000


### Inferência via script

make infer


### Subir a API FastAPI

make api


Docs interativas: `http://127.0.0.1:8000/docs`

**POST `/predict`** – exemplo de body:

{
"length1": 23.2,
"length2": 25.4,
"length3": 30.0,
"height": 11.52,
"width": 4.02
}


Resposta:

{
"predicted_weight": 312.85
}


**POST `/predict-image`** – exemplo de chamada:

- Método: `POST`  
- URL: `http://localhost:8000/predict-image?quantity=3`  
- Body: arquivo de imagem (`multipart/form-data`, campo `file`).

Resposta (exemplo):

{
"image_width_px": 360,
"image_height_px": 360,
"features_used": {
"Length1": 36,
"Length2": 40,
"Length3": 45,
"Height": 36,
"Width": 18
},
"predicted_weight": 1560.34,
"quantity": 3,
"biomass_kg": 4.68
}


### App Streamlit

Com a API rodando em `localhost:8000`:

python -m streamlit run app_streamlit.py


Acessar: `http://localhost:8501`.

- Aba **“Medidas manuais”**: preencha as medidas e clique em “Prever peso (medidas)”.  
- Aba **“Imagem do peixe”**:  
  - faça upload de uma foto de peixe;  
  - informe a quantidade de peixes no tanque;  
  - clique em “Calcular medidas, peso e biomassa” para ver:
    - dimensões da imagem,  
    - medidas derivadas,  
    - peso previsto individual,  
    - biomassa estimada do tanque.

### Testes

make test

ou simplesmente

pytest


### Docker

make docker-build
make docker-run


API disponível em `http://localhost:8000`.

## CI/CD com GitHub Actions

O repositório contém um workflow em `.github/workflows/ci.yml` que executa:

- Checkout do código.  
- Configuração do Python 3.11.  
- Instalação das dependências (`pip install -r requirements.txt`).  
- Execução dos testes com `pytest`.

O pipeline roda automaticamente em:

- Todo **push** para a branch `main`.  
- Toda **pull request** aberta para `main`.

Isso garante que o projeto sempre builda e passa nos testes em um ambiente limpo.

## Tecnologias

- Python  
- Scikit-Learn  
- Pandas, NumPy  
- FastAPI, Uvicorn  
- Streamlit  
- MLflow  
- Pytest  
- Docker  
- Makefile  
- GitHub Actions

## Práticas de MLOps

- Separação de etapas: preparação (`data_prep`), treinamento (`train`), inferência (`infer`).  
- Métrica de avaliação explícita (**MAE**).  
- Tracking de experimentos com **MLflow** (parâmetros, métricas, modelo).  
- Testes unitários da função de inferência e teste de integração da API.  
- Deploy da API em **Docker**.  
- **Makefile** para padronizar comandos.  
- App **Streamlit** para:
  - entrada manual de medidas;  
  - envio de imagem do peixe, com extração de dimensões, predição de peso individual e **cálculo de biomassa do tanque**.  
- Endpoint `/predict-image` especializado para fluxo por imagem + biomassa, simulando uso de câmeras e visão computacional.  
- **CI com GitHub Actions**, rodando instalação e testes a cada push/PR.

## Possíveis melhorias

- Experimentar outros modelos (RandomForest, XGBoost etc.) e comparar via MLflow.  
- Mais testes (unitários e integração).  
- Integrar validação de dados / monitoramento de drift (EvidentlyAI ou similar).  
- Pipeline de CD para deploy automático em ambiente cloud.  
- Deploy em AWS Lambda ou outro ambiente serverless.