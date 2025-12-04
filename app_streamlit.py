from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = "http://localhost:8000"
LOG_PATH = Path("data") / "log_predictions.csv"


def _call_predict_image(files, params):
    """Chama a API /predict-image e retorna o JSON (sem desenhar nada aqui)."""
    resp = requests.post(
        f"{API_URL}/predict-image",
        files=files,
        params=params,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def _get_largest_contour_bbox(image_rgb: np.ndarray) -> tuple[int, int, int, int]:
    """
    Recebe uma imagem RGB (numpy) e retorna (x, y, w, h) do maior contorno.
    Se não achar nada, usa o tamanho da imagem inteira.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        h, w = gray.shape
        return 0, 0, w, h

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def main():
    st.title("Predição de peso de peixes")

    tab_manual, tab_image, tab_dash = st.tabs(
        ["Medidas manuais", "Imagem do peixe", "Dashboard"]
    )

    # ---------------- Aba 1: medidas manuais ----------------
    with tab_manual:
        st.subheader("Entrada manual de medidas")

        length1 = st.number_input("Length1", min_value=0.0, value=23.2)
        length2 = st.number_input("Length2", min_value=0.0, value=25.4)
        length3 = st.number_input("Length3", min_value=0.0, value=30.0)
        height = st.number_input("Height", min_value=0.0, value=11.52)
        width = st.number_input("Width", min_value=0.0, value=4.02)
        tank_id_manual = st.text_input("ID do tanque / lote", value="manual_tank")

        if st.button("Prever peso (medidas)"):
            payload = {
                "length1": length1,
                "length2": length2,
                "length3": length3,
                "height": height,
                "width": width,
            }
            params = {"tank_id": tank_id_manual}
            try:
                resp = requests.post(
                    f"{API_URL}/predict", json=payload, params=params, timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Peso previsto: {data['predicted_weight']:.2f} g "
                        f"(tanque = {data['tank_id']})"
                    )
                else:
                    st.error(f"Erro ao chamar API: {resp.status_code}")
            except Exception as e:
                st.error(f"Erro de conexão com API: {e}")

    # ---------------- Aba 2: imagem + biomassa (upload + câmera) ----------------
    with tab_image:
        st.subheader("Imagem do peixe (upload ou câmera)")

        quantity = st.number_input(
            "Quantidade de peixes no tanque", min_value=1, value=1, step=1
        )
        tank_id_image = st.text_input("ID do tanque / lote (imagem)", value="tank_1")

        mode = st.radio(
            "Fonte da imagem",
            ["Upload de arquivo", "Câmera (webcam)"],
            horizontal=True,
        )

        # --- Modo upload ---
        if mode == "Upload de arquivo":
            uploaded_file = st.file_uploader(
                "Envie a foto do peixe", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Imagem enviada", use_container_width=True)

                if st.button("Calcular e mostrar contorno (upload)"):
                    # envia para API
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    params = {"quantity": int(quantity), "tank_id": tank_id_image}

                    try:
                        data = _call_predict_image(files, params)
                    except Exception as e:
                        st.error(f"Erro na API: {e}")
                    else:
                        # desenhar contorno e texto na imagem
                        np_img = np.array(img)
                        x, y, w, h = _get_largest_contour_bbox(np_img)

                        draw = ImageDraw.Draw(img)
                        # retângulo do "peixe"
                        draw.rectangle(
                            [x, y, x + w, y + h],
                            outline=(255, 0, 0),
                            width=3,
                        )
                        text = (
                            f"{w}x{h} px\n"
                            f"Peso: {data['predicted_weight']:.1f} g\n"
                            f"Biomassa: {data['biomass_kg']:.2f} kg"
                        )
                        # caixinha de fundo
                        draw.rectangle(
                            [x, y - 50 if y - 50 > 0 else 0, x + 220, y],
                            fill=(0, 0, 0, 180),
                        )
                        draw.text((x + 5, max(0, y - 45)), text, fill=(255, 255, 255))

                        st.image(
                            img,
                            caption="Imagem com contorno e medidas detectadas",
                            use_container_width=True,
                        )

                        st.write("Medidas derivadas usadas no modelo:")
                        st.json(data["features_used"])

        # --- Modo câmera com preview e clique para foto ---
        if mode == "Câmera (webcam)":
            st.info("Aponte a câmera para o peixe e clique em 'Tirar foto e calcular'.")

            camera_image = st.camera_input("Pré-visualização da câmera")

            if camera_image is not None:
                img = Image.open(camera_image).convert("RGB")

                if st.button("Tirar foto, calcular e mostrar contorno"):
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)

                    files = {"file": ("camera.png", buf.getvalue(), "image/png")}
                    params = {"quantity": int(quantity), "tank_id": tank_id_image}

                    try:
                        data = _call_predict_image(files, params)
                    except Exception as e:
                        st.error(f"Erro na API: {e}")
                    else:
                        # desenhar contorno e texto na imagem capturada
                        np_img = np.array(img)
                        x, y, w, h = _get_largest_contour_bbox(np_img)

                        draw = ImageDraw.Draw(img)
                        draw.rectangle(
                            [x, y, x + w, y + h],
                            outline=(0, 255, 0),
                            width=3,
                        )
                        text = (
                            f"{w}x{h} px\n"
                            f"Peso: {data['predicted_weight']:.1f} g\n"
                            f"Biomassa: {data['biomass_kg']:.2f} kg"
                        )
                        draw.rectangle(
                            [x, y - 50 if y - 50 > 0 else 0, x + 220, y],
                            fill=(0, 0, 0, 180),
                        )
                        draw.text((x + 5, max(0, y - 45)), text, fill=(255, 255, 255))

                        st.image(
                            img,
                            caption="Foto com contorno e medidas detectadas",
                            use_container_width=True,
                        )

                        st.write("Medidas derivadas usadas no modelo:")
                        st.json(data["features_used"])

    # ---------------- Aba 3: dashboard ----------------
    with tab_dash:
        st.subheader("Dashboard de previsões")

        if LOG_PATH.exists():
            df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
            df = df.sort_values("timestamp")

            # Filtros
            sources = df["source"].unique().tolist()
            selected_sources = st.multiselect(
                "Fonte de dados (source)", sources, default=sources
            )

            tanks = df["tank_id"].unique().tolist()
            selected_tanks = st.multiselect(
                "Tanques / lotes", tanks, default=tanks
            )

            min_date = df["timestamp"].min().date()
            max_date = df["timestamp"].max().date()
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Data inicial", value=min_date)
            with col2:
                end_date = st.date_input("Data final", value=max_date)

            mask = (
                df["source"].isin(selected_sources)
                & df["tank_id"].isin(selected_tanks)
                & (df["timestamp"].dt.date >= start_date)
                & (df["timestamp"].dt.date <= end_date)
            )
            df_filt = df[mask]

            st.write("Últimas previsões filtradas:")
            st.dataframe(df_filt.tail(20))

            if not df_filt.empty:
                st.subheader("Biomassa estimada ao longo do tempo (kg)")
                st.line_chart(
                    df_filt.set_index("timestamp")["biomass_kg"],
                    use_container_width=True,
                )

                st.subheader("Distribuição de peso previsto (g)")
                st.bar_chart(
                    df_filt[["predicted_weight_g"]],
                    use_container_width=True,
                )
            else:
                st.info("Nenhum dado para os filtros selecionados.")
        else:
            st.info("Ainda não há previsões registradas para gerar gráficos.")


if __name__ == "__main__":
    main()
