from app.graph import FabricInspectionGraph

class InspectionService:

    def __init__(self):
        self.graph = FabricInspectionGraph(
            model_path="models/best_lora_dino.pt",
            annoy_path="data/fabric.ann",
            annoy_index_info="data/annoy_index_info.json",
        )

    def inspect(
        self,
        image_path: str,
        user_question: str
    ):
        return self.graph.run(
            image_path=image_path,
            user_question=user_question,
        )