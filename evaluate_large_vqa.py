import gc
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "vqa_data"
IMAGES_DIR = DATA_DIR / "val2014"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

VAL2014_ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"
QUESTIONS_ZIP_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
ANNOTATIONS_ZIP_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"

QUESTIONS_JSON_NAME = "v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_JSON_NAME = "v2_mscoco_val2014_annotations.json"

NUM_SAMPLES = 30000
RESULTS_CSV = "vqa_project_results.csv"
CHUNK_SIZE = 8 * 1024 * 1024


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Zaten mevcut, atlaniyor: {dest.name}")
        return

    part_path = dest.with_suffix(dest.suffix + ".part")
    resume_pos = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}

    print(f"Indiriliyor: {url}")
    try:
        with requests.get(url, stream=True, headers=headers, timeout=120) as response:
            if response.status_code == 416:
                resume_pos = 0
                part_path.unlink(missing_ok=True)
                with requests.get(url, stream=True, timeout=120) as retry_response:
                    retry_response.raise_for_status()
                    total = int(retry_response.headers.get("content-length", 0))
                    _write_chunks(retry_response, part_path, 0, total)
            else:
                response.raise_for_status()
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    total = int(content_range.split("/")[-1]) if "/" in content_range else 0
                else:
                    resume_pos = 0
                    part_path.unlink(missing_ok=True)
                    total = int(response.headers.get("content-length", 0))
                _write_chunks(response, part_path, resume_pos, total)
    except Exception as e:
        print(f"Indirme hatasi ({dest.name}): {e}")
        sys.exit(1)

    part_path.rename(dest)
    print(f"Indirme tamamlandi: {dest.name}")


def _write_chunks(response, part_path: Path, resume_pos: int, total: int) -> None:
    mode = "ab" if resume_pos else "wb"
    downloaded = resume_pos
    with open(part_path, mode) as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                print(f"\r  Ilerleme: {pct:.1f}% ({downloaded // (1024 * 1024)} MB)", end="")
    print()


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"Zaten acilmis, atlaniyor: {extract_dir.name}")
        return

    print(f"Zip aciliyor: {zip_path.name}")
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        marker.touch()
    except Exception as e:
        print(f"Zip acma hatasi ({zip_path.name}): {e}")
        sys.exit(1)


def find_file(root: Path, filename: str) -> Path:
    for path in root.rglob(filename):
        return path
    raise FileNotFoundError(f"{filename} bulunamadi: {root}")


def prepare_coco_images() -> None:
    zip_path = DATA_DIR / "val2014.zip"
    download_file(VAL2014_ZIP_URL, zip_path)
    extract_zip(zip_path, DATA_DIR)
    if not IMAGES_DIR.exists():
        nested = DATA_DIR / "val2014"
        if nested.exists() and any(nested.glob("*.jpg")):
            pass
        else:
            print(f"Gorsel klasoru bulunamadi: {IMAGES_DIR}")
            sys.exit(1)


def prepare_vqa_annotations() -> tuple[Path, Path]:
    questions_zip = DATA_DIR / "v2_Questions_Val_mscoco.zip"
    annotations_zip = DATA_DIR / "v2_Annotations_Val_mscoco.zip"

    download_file(QUESTIONS_ZIP_URL, questions_zip)
    download_file(ANNOTATIONS_ZIP_URL, annotations_zip)

    questions_extract = DATA_DIR / "questions_val"
    annotations_extract = DATA_DIR / "annotations_val"
    extract_zip(questions_zip, questions_extract)
    extract_zip(annotations_zip, annotations_extract)

    questions_json = find_file(questions_extract, QUESTIONS_JSON_NAME)
    annotations_json = find_file(annotations_extract, ANNOTATIONS_JSON_NAME)
    return questions_json, annotations_json


def load_validation_samples(limit: int) -> list[dict]:
    questions_path, annotations_path = prepare_vqa_annotations()

    with open(questions_path, encoding="utf-8") as f:
        questions_data = json.load(f)
    with open(annotations_path, encoding="utf-8") as f:
        annotations_data = json.load(f)

    annotations_by_qid = {
        ann["question_id"]: ann for ann in annotations_data["annotations"]
    }

    samples = []
    for question in questions_data["questions"]:
        qid = question["question_id"]
        annotation = annotations_by_qid.get(qid)
        if annotation is None:
            continue

        samples.append(
            {
                "image_id": question["image_id"],
                "question_id": qid,
                "question": question["question"],
                "answers": [a["answer"] for a in annotation["answers"]],
            }
        )
        if len(samples) >= limit:
            break

    return samples


def image_path_for_id(image_id: int) -> Path:
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    path = IMAGES_DIR / filename
    if path.exists():
        return path
    alt = DATA_DIR / "val2014" / filename
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Gorsel bulunamadi: {filename}")


def extract_ground_truths(answers: list[str]) -> list[str]:
    return answers


def is_correct(predicted: str, ground_truths: list[str]) -> bool:
    predicted_norm = predicted.strip().lower()
    return any(predicted_norm == gt.strip().lower() for gt in ground_truths)


def main() -> None:
    device = "cuda"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Adim 1: COCO val2014 gorselleri ===")
    prepare_coco_images()

    print("\n=== Adim 2: VQA v2 soru ve annotasyonlari ===")
    samples = load_validation_samples(NUM_SAMPLES)
    print(f"Yuklenen ornek sayisi: {len(samples)}")

    print("\n=== Adim 3: BLIP modeli ===")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    model.eval()

    results = []
    correct_count = 0

    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            try:
                image = Image.open(image_path_for_id(sample["image_id"])).convert("RGB")
            except FileNotFoundError as e:
                print(f"Uyari: {e} — ornek atlaniyor.")
                continue

            question = sample["question"]
            ground_truths = extract_ground_truths(sample["answers"])

            inputs = processor(image, question, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs)
            predicted = processor.decode(output_ids[0], skip_special_tokens=True)

            correct = is_correct(predicted, ground_truths)
            if correct:
                correct_count += 1

            results.append(
                {
                    "MS COCO Image ID": sample["image_id"],
                    "Question ID": sample["question_id"],
                    "Soru": question,
                    "Yapay Zeka Cevabi": predicted,
                    "Gercek Cevaplar": "; ".join(ground_truths),
                    "Durum": "Dogru" if correct else "Yanlis",
                }
            )

            image.close()
            del image, inputs, output_ids
            torch.cuda.empty_cache()
            gc.collect()

            print(f"Isleniyor: {index} / {NUM_SAMPLES}")

    if not results:
        print("Hic ornek islenemedi.")
        sys.exit(1)

    accuracy = (correct_count / len(results)) * 100
    df = pd.DataFrame(results)
    df.to_csv(PROJECT_DIR / RESULTS_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSonuclar kaydedildi: {RESULTS_CSV}")
    print(f"Nihai Model Dogruluk Orani (Accuracy): %{accuracy:.2f}")


if __name__ == "__main__":
    main()
