from pathlib import Path
from datetime import datetime
import csv
import io

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
from PIL import Image

from src.infer import predict_weight


def get_largest_contour_bbox(image: np.ndarray) -> tuple[int, int]:
    """
    Recebe uma imagem RGB (array) e retorna (width, height) do melhor contorno.
    Critérios:
      - área mínima (descarta contornos muito pequenos)
      - proporção largura/altura (prefere contornos alongados)
    Se nada for encontrado, usa o tamanho da imagem inteira como fallback.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = gray.shape
    img_area = w_img * h_img

    best_bbox = None
    best_score = -1.0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 0.05 * img_area:  # descarta contorno com área < 5% da imagem
            continue

        aspect = max(w, h) / max(1, min(w, h))  # razão de aspecto
        if aspect < 1.5:  # descarta contornos pouco alongados
            continue

        # score simples: área * alongamento
        score = area * aspect
        if score > best_score:
            best_score = score
            best_bbox = (w, h)

    if best_bbox is not None:
        return best_bbox

    # fallback: sem contorno bom, usa a imagem inteira
    return w_img, h_img



app = FastAPI()

LOG_PATH = Path("data") / "log_predictions.csv"


def log_prediction(
    source: str,
    predicted_weight: float,
    quantity: int,
    biomass_kg: float,
    tank_id: str,
) -> None:
    """Registra previsões em CSV para uso no dashboard."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "source",
                    "tank_id",
                    "predicted_weight_g",
                    "quantity",
                    "biomass_kg",
                ]
            )
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                source,
                tank_id,
                float(predicted_weight),
                int(quantity),
                float(biomass_kg),
            ]
        )


@app.get("/")
def read_root():
    return {"status": "ok"}


class PredictRequest(BaseModel):
    length1: float
    length2: float
    length3: float
    height: float
    width: float


@app.post("/predict")
def predict(
    request: PredictRequest,
    tank_id: str = Query("manual_tank", description="Identificador do tanque/lote"),
):
    """Predição de peso a partir de medidas manuais."""
    weight = predict_weight(
        request.length1,
        request.length2,
        request.length3,
        request.height,
        request.width,
    )

    biomass_kg = weight / 1000.0
    log_prediction(
        source="manual",
        predicted_weight=weight,
        quantity=1,
        biomass_kg=biomass_kg,
        tank_id=tank_id,
    )

    return {"predicted_weight": weight, "tank_id": tank_id}


@app.post("/predict-image")
async def predict_from_image(
    file: UploadFile = File(...),
    quantity: int = Query(1, ge=1, description="Quantidade de peixes no tanque"),
    tank_id: str = Query("tank_1", description="Identificador do tanque/lote"),
):
    """Predição de peso e biomassa a partir de imagem do peixe."""
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # converte PIL -> numpy (RGB)
    np_image = np.array(pil_image)

    # pega bounding box do maior contorno (supostamente o peixe)
    width_px, height_px = get_largest_contour_bbox(np_image)

    # mapeamento simples de pixels -> medidas (mock de visão computacional),
    # agora usando só o tamanho do maior contorno
    length1 = width_px / 10
    length2 = width_px / 9
    length3 = width_px / 8
    height = height_px / 10
    fish_width = width_px / 20

    predicted_weight = predict_weight(
        length1, length2, length3, height, fish_width
    )

    biomass_kg = (predicted_weight * quantity) / 1000.0

    log_prediction(
        source="image",
        predicted_weight=predicted_weight,
        quantity=quantity,
        biomass_kg=biomass_kg,
        tank_id=tank_id,
    )

    return {
        "image_width_px": width_px,
        "image_height_px": height_px,
        "features_used": {
            "Length1": length1,
            "Length2": length2,
            "Length3": length3,
            "Height": height,
            "Width": fish_width,
        },
        "predicted_weight": predicted_weight,
        "quantity": quantity,
        "biomass_kg": biomass_kg,
        "tank_id": tank_id,
    }
