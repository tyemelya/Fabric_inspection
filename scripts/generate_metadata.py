from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from app.ml.data import get_data_loader
from tqdm.auto import tqdm
from PIL import Image
import torch
import json
import re
import sqlite3

MAX_SIZE = (640, 640)

def parse_json_response(response: str) -> dict:
    # Remove ```json and ```
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match is None:
        raise ValueError("No JSON found.")

    return json.loads(match.group())

def main():
    connection = sqlite3.connect("data/fabric.db")
    cur = connection.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fabric_defects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT UNIQUE,
        defect_class TEXT,
        severity TEXT,
        description TEXT
    )
    """)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map=device
    )

    _, dataset = get_data_loader("data/fabric_dataset", batch_size=32)
    defectfree = 4

    for processed, (image_path, label) in enumerate(
        tqdm(dataset.samples, desc="Generating metadata", unit="image"),
        start=2101
    ):
        # Check if image already exists in database
        cur.execute(
            """
            SELECT 1 FROM fabric_defects
            WHERE image_path = ?
            LIMIT 1
            """,
            (image_path,)
        )

        if cur.fetchone() is not None:
            continue  # skip generation
        
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            if image.width > MAX_SIZE[0] or image.height > MAX_SIZE[1]:
                print(
                    f"Resizing {image_path}: "
                    f"{image.width}x{image.height} -> "
                    f"max {MAX_SIZE}"
                )
                image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
        
            class_name = dataset.classes[label]
        
            if label==defectfree:
                cur.execute(
                        """
                            INSERT OR IGNORE INTO fabric_defects
                            (image_path, defect_class, severity, description)
                            VALUES (?, ?, ?, ?)
                        """,
                            (
                                image_path,
                                class_name,
                                "",
                                "",
                            ),
                        )
                continue
        
            prompt = f"""
                The ground-truth defect category is "{class_name}".
                Do NOT classify the defect again.
                Instead, analyze only the visual appearance.
                Return ONLY valid JSON.
                {{
                    "defect_type": "{class_name}",
                    "severity": "low | medium | high",
                    "description": ""
                }}
            """

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]
        
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(model.device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]

            metadata = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
    
            try:
                metadata = parse_json_response(metadata)
            except Exception as e:
                print(f"Failed to parse {image_path}: {e}")
                continue
            
            cur.execute(
            """
                INSERT OR IGNORE INTO fabric_defects
                (image_path, defect_class, severity, description)
                VALUES (?, ?, ?, ?)
            """,
                (
                    image_path.replace("fabric_dataset2", "fabric_dataset"),
                    class_name,
                    metadata["severity"],
                    metadata["description"],
                ),
            )
            if processed % 10 == 0:
                connection.commit()

    connection.commit()
    connection.close()

if __name__ == "__main__":
    main()