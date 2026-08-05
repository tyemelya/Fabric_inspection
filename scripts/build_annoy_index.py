from annoy import AnnoyIndex
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import sqlite3
from app.tools.vision import VisionTool
import json

NUMBER_OF_TREES = 25

def build_annoy_index(vtool, res, n_trees=25):
    _, first_image, _ = res[0]
    first_embedding = vtool.get_embedding(first_image)
    feature_dim = len(first_embedding)
    
    index = AnnoyIndex(feature_dim, "angular")
    vtool.model.eval()

    pbar = tqdm(res, leave=False, desc="Indexing")
    
    for idx, image_path, _ in pbar:
        feats = vtool.get_embedding(image_path)
        index.add_item(idx, feats)

    index.build(n_trees)
    return index, feature_dim

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    connection = sqlite3.connect("data/fabric.db")
    cur = connection.cursor()
    cur.execute(
             """
                 SELECT id, image_path, defect_class FROM fabric_defects
             """
    )
    res = cur.fetchall()
    connection.close()

    vtool = VisionTool(model_path="models/best_lora_dino.pt", device=device)
    index, feature_dim = build_annoy_index(vtool, res, n_trees=NUMBER_OF_TREES)
    index.save("./data/fabric.ann")
    
    with open("data/annoy_index_info.json", "w") as f:
        json.dump({
                    "feature_dim": feature_dim,
                    "metric": "angular",
                    "number_of_trees": NUMBER_OF_TREES,
                    "index_file": "fabric.ann"
                }, f, indent=2)

if __name__ == "__main__":
    main()
