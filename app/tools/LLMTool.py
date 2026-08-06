from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
from dataclasses import dataclass

@dataclass
class RoutingDecision:
    use_retrieval: bool
    reason: str

def parse_json_response(response: str) -> dict:
    # Remove ```json and ```
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match is None:
        raise ValueError("No JSON found.")

    return json.loads(match.group())

class LLMTool:
    def __init__(self, model_name, device=None):
        self.device = (
            device 
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=self.device
        )

        self.model.eval()

    def generate(self, prompt, max_new_tokens=256):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generated = outputs[:, inputs.input_ids.shape[1]:]

        return self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ) 
    
    def plan_workflow(self, user_question):
        prompt = f"""
            You are planning a fabric inspection workflow.

            Available tools:
            - VisionTool: classifies the fabric defect.
            - Retrieval: retrieves similar historical defect cases and supporting evidence.

            If the user asks for similar examples, comparison, justification,
            or explanation, set "use_retrieval" to true.

            Otherwise set it to false.

            Return ONLY valid JSON.

            {{
                "use_retrieval": true,
                "reason": ""
            }}

            User question:
            {user_question}
        """        

        try:
            decision = parse_json_response(self.generate(prompt))
        except Exception as e:
            raise ValueError(f"Invalid routing decision: {e}")

        return RoutingDecision(**decision)