import gc
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = PROJECT_DIR / "vqa_project_results.csv"
MODEL_ID = "Salesforce/blip-vqa-base"


@st.cache_resource
def load_vqa_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model, device


@st.cache_data
def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig")
    status_map = {
        "Dogru": "Correct",
        "Yanlis": "Incorrect",
        "Correct": "Correct",
        "Incorrect": "Incorrect",
    }
    status_col = "Durum" if "Durum" in df.columns else "Status"
    df["Result"] = df[status_col].astype(str).str.strip().map(status_map).fillna(df[status_col])
    return df


def run_inference(image: Image.Image, question: str) -> str:
    processor, model, device = load_vqa_model()
    rgb_image = image.convert("RGB")

    with torch.inference_mode():
        inputs = processor(rgb_image, question, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs)
        answer = processor.decode(output_ids[0], skip_special_tokens=True)

    del inputs, output_ids
    torch.cuda.empty_cache()
    gc.collect()
    return answer


def render_live_vqa_tab() -> None:
    st.header("Live Visual Question Answering (VQA)")
    st.write(
        "Upload an image and ask a natural-language question. "
        "The BLIP VQA model will analyze the image on your GPU and return an answer."
    )

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    question = st.text_input(
        "Ask a question about the image:",
        placeholder="e.g., What color is the car?",
    )
    st.caption("Examples: 'What color is the...?', 'How many... are there?'")

    if st.button("Run Model & Answer", type="primary"):
        if uploaded is None:
            st.warning("Please upload an image before running the model.")
            return
        if not question.strip():
            st.warning("Please enter a question about the image.")
            return

        with st.spinner("Running BLIP VQA on GPU..."):
            try:
                answer = run_inference(image, question.strip())
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

        st.success(f"**Model Answer:** {answer}")


def render_analysis_tab() -> None:
    st.header("30,000 Test Set Analysis & Error Reporting")

    if not RESULTS_CSV.exists():
        st.error(f"Results file not found: `{RESULTS_CSV.name}`")
        st.info("Run `python evaluate_large_vqa.py` first to generate evaluation results.")
        return

    df = load_results()
    total_samples = len(df)
    correct_count = (df["Result"] == "Correct").sum()
    accuracy = (correct_count / total_samples * 100) if total_samples else 0.0

    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", f"{accuracy:.2f}%")
    col2.metric("Total Test Samples", f"{total_samples:,}")

    counts = df["Result"].value_counts().reindex(["Correct", "Incorrect"], fill_value=0)
    pie_df = counts.reset_index()
    pie_df.columns = ["Result", "Count"]

    fig = px.pie(
        pie_df,
        names="Result",
        values="Count",
        title="Prediction Outcomes on the Test Set",
        color="Result",
        color_discrete_map={"Correct": "#2ecc71", "Incorrect": "#e74c3c"},
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

    show_incorrect_only = st.checkbox("Show Incorrect Predictions Only")
    display_df = df[df["Result"] == "Incorrect"] if show_incorrect_only else df
    st.dataframe(display_df, use_container_width=True, height=500)


def main() -> None:
    st.set_page_config(
        page_title="VQA Project Dashboard",
        page_icon="🖼️",
        layout="wide",
    )

    st.title("Visual Question Answering (VQA) Project")
    st.markdown(
        "Interactive demo and evaluation dashboard for the **Salesforce/blip-vqa-base** model "
        "on MS COCO / VQA v2 data."
    )

    tab_live, tab_analysis = st.tabs(
        [
            "Live Visual Question Answering (VQA)",
            "30,000 Test Set Analysis & Error Reporting",
        ]
    )

    with tab_live:
        render_live_vqa_tab()

    with tab_analysis:
        render_analysis_tab()


if __name__ == "__main__":
    main()
