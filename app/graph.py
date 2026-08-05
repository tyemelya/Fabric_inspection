from langgraph.graph import StateGraph, START, END
from app.state import InspectionState
import torch
from app.tools.vision import VisionTool
from app.tools.similarity import SimilarityTool
from app.tools.metadata import MetadataTool
from app.tools.evidence import EvidenceTool
from app.tools.LLMTool import LLMTool
from dataclasses import dataclass

@dataclass
class RoutingDecision:
    use_retrieval: bool
    reason: str

class FabricInspectionGraph:
    def __init__(
        self,
        model_path,
        annoy_path,
        annoy_index_info,
        device=None
    ):
        self.device = (
            device 
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.llm_tool = LLMTool(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            device=self.device
        )

        self.vtool = VisionTool(
            model_path=model_path,
            device=self.device
        )

        self.metadata_tool = MetadataTool()

        self.similarity_tool = SimilarityTool(
            annoy_path=annoy_path,
            annoy_index_info=annoy_index_info,
            metadata_tool=self.metadata_tool
        )

        self.evidence_tool = EvidenceTool()

        self.graph = self._build_graph()

    def router_node(self, state):
        decision =self.llm_tool.plan_workflow(state["user_question"])
        
        return {
            "routing": decision
        }

    def vision_node(self, state):
        result = self.vtool.analyze(
            image_path=state["image_path"]
        )

        return {
            "vision_result": result
        }


    def similarity_node(self, state):
        retrieved = self.similarity_tool.search(
            state["vision_result"].embedding
        )

        return {
            "retrieved_cases": retrieved
        }


    def evidence_node(self, state):
        result = self.evidence_tool.get_evidence(
            prediction=state["vision_result"].prediction,
            retrieved_cases=state["retrieved_cases"]
        )

        return {
            "evidence": result
        }

    def report_node(self, state):
        prompt = f"""
        You are a textile inspection assistant.

        Image analysis result:
        - Defect: {state["vision_result"].prediction}
        - Vision confidence: {state["vision_result"].confidence}

        Evidence:
        {state.get("evidence", "No retrieval evidence available")}

        Retrieved examples:
        {state.get("retrieved_cases", [])}

        User question:
        {state["user_question"]}

        Answer clearly and briefly.
        """

        response = self.llm_tool.generate(prompt)

        return {
            "report": response
        }
    
    def _route(self, state):
        return (
            "retrieval"
            if state["routing"].use_retrieval
            else "vision_only"
        )   
    
    def _build_graph(self):

        graph = StateGraph(InspectionState)
        graph.add_node("router", self.router_node)
        graph.add_node("vision", self.vision_node)
        graph.add_node("similarity", self.similarity_node)
        graph.add_node("evidence", self.evidence_node)
        graph.add_node("report", self.report_node)
        
        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            self._route,
            {
                "vision_only": "vision",
                "retrieval": "vision",
            },
        )
        
        graph.add_conditional_edges(
            "vision",
            self._route,
            {
                "vision_only": "report",
                "retrieval": "similarity",
            },  
        )
        
        graph.add_edge("similarity", "evidence")
        graph.add_edge("evidence", "report")
        graph.add_edge("report", END)
        
        return graph.compile()

    def run(self, image_path, user_question):
        return self.graph.invoke(
            {
                "image_path": image_path,
                "user_question": user_question
            }
        )

    def save_graph_image(self, path):

        png_bytes = self.graph.get_graph().draw_mermaid_png()

        with open(path, "wb") as f:
            f.write(png_bytes)