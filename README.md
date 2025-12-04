# Predição de peso e biomassa de peixes

## Problema

Estimar:

- peso individual de peixes (g) a partir de medidas morfométricas;
- biomassa total de um tanque (kg) = peso médio × quantidade.

O foco é demonstrar um pipeline de MLOps (treino → API → app → logs → dashboard), aproximando o cenário de uso real da empresa (câmera + biomassa em tempo quase real).

---

## Arquitetura da solução

![Arquitetura da solução – Fish Weight MLOps](./docs/arquitetura-mlops.jpg)

- **Modelo de ML (tabular)**  
  - Regressão treinada em dataset de peixes com as features  
    `Length1`, `Length2`, `Length3`, `Height`, `Width`.  
  - Saída: peso previsto em gramas.

- **API (FastAPI)**  
  - `POST /predict`: recebe medidas manuais e retorna peso.  
  - `POST /predict-image`: recebe uma imagem, aplica um mock simples de visão (contornos via OpenCV) para extrair largura/altura em pixels, gera as 5 features, calcula peso e biomassa e registra logs em `data/log_predictions.csv`.

- **App Streamlit**  
  - Aba **Medidas manuais**: formulário para envio ao endpoint `/predict`.  
  - Aba **Imagem do peixe**: upload/webcam → chama `/predict-image` → exibe a foto com o retângulo detectado + texto de peso/biomassa sobre a imagem.  
  - Aba **Dashboard**: lê o CSV de logs, mostra tabela das últimas previsões e gráficos de biomassa ao longo do tempo e distribuição de peso.

---

## Screenshots

### App Streamlit – imagem do peixe

![Tela de previsão por imagem](./docs/streamlit-image.png)

### Dashboard de biomassa

![Dashboard de biomassa](./docs/dashboard.png)

---

## Como executar o treinamento

O script de treino (ajuste o nome conforme seu repo, por exemplo `src/train_model.py`):

python src/train_model.py


Esse script deve:

- ler o dataset tabular;
- treinar o modelo de regressão;
- salvar o artefato em `models/model.pkl` (carregado por `src/infer.py`).

---

## Como realizar a inferência

### Subir a API

uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000


### Testar via Streamlit


- Aba **Medidas manuais** → envia JSON para `/predict`.  
- Aba **Imagem do peixe** → envia imagem + parâmetros para `/predict-image` e mostra o contorno usado para o cálculo.

Também é possível chamar a API diretamente via HTTP (curl, Postman etc.) enviando JSON ou form-data de imagem.

---

## Dependências principais

- Python 3.10+
- `fastapi`, `uvicorn`, `pydantic`
- `numpy`, `pandas`, `scikit-learn` (ou lib usada no modelo)
- `Pillow`, `opencv-python`
- `requests`, `streamlit`

Instalação (exemplo):

pip install fastapi uvicorn pydantic numpy pandas scikit-learn pillow opencv-python requests streamlit


---

## Possíveis melhorias

- Substituir o mock de visão (contornos) por um modelo de detecção/segmentação de peixes treinado em dataset anotado (ex.: YOLO, Mask R‑CNN).  
- Calibrar pixels → centímetros usando referência física na cena.  
- Especializar modelos por espécie/tipo de tanque.  
- Monitorar métricas de modelo e API (latência, erro, drift) em um painel de MLOps.
