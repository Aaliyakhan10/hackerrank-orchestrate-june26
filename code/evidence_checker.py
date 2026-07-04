import os
import json
from typing import Dict, Any, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

class EvidenceOutput(BaseModel):
    object_visible: bool
    object_mismatch: bool = False
    part_visible: bool
    damage_visible: bool
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    valid_image: bool

class EvidenceChecker:
    def __init__(self, model_name="google/gemma-4-31b-it:free"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model_name = model_name

    def format_image_evidence(self, image_packet: List[Dict[str, Any]]) -> str:
        """
        Formats the output of Agent 2 (Observation Graph) into a clean readable string.
        """
        text = ""
        for i, img in enumerate(image_packet):
            path = img.get('image_path', f'Unknown_{i}')
            text += f"Image {i+1} ({path}):\n"
            text += f"- Object: {img.get('object_class')}\n"
            text += f"- Parts Visible: {img.get('parts_visible')}\n"
            text += f"- Damage Detected: {json.dumps(img.get('damage_detected', []))}\n"
            text += f"- Quality Flags: {img.get('quality_flags', [])}\n\n"
        return text.strip()

    def verify_evidence(self, base_inputs: Dict[str, Any], claim_packet: Dict[str, Any], image_packet: List[Dict[str, Any]], req_text: str, retrieval_packet: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Agent 4 checks if the evidence meets the minimal standard AND evaluates retrieval memory context.
        """
        system_prompt = """You are an evidence standard evaluator for damage claims.

Your job: decide whether a submitted set of images satisfies the 
minimum evidence requirement for this claim type.

You must output ONLY valid JSON, no prose:
You must explicitly answer the coverage questions and return ONLY valid JSON:
{
  "object_visible": true | false,
  "object_mismatch": true | false,
  "part_visible": true | false,
  "damage_visible": true | false,
  "evidence_standard_met": true | false,
  "evidence_standard_met_reason": "one concise sentence",
  "valid_image": true | false
}

Rules:
- 'object_visible' is true if the base object (e.g. car or box) is seen in ANY image.
- 'object_mismatch' is true ONLY if the image clearly shows a completely different object than what is claimed (e.g., a soda can instead of a shipping box).
- 'part_visible' is true if the claimed object_part (or a reasonable synonym like tape, flap, edge, seal, etc.) is seen in ANY image.
- 'damage_visible' is true if the requested damage (or a related condition) is seen in ANY image. Ensure the `damage_detected` type semantically matches the `issue_type`. If the user claims a stain but the vision model detects a burn mark or hole, `damage_visible` MUST be false.
- 'evidence_standard_met' is ONLY true if the requirement text is satisfied. Do NOT automatically fail this just because the exact word for 'part' or 'damage' isn't explicitly listed; use visual reasoning.
- 'valid_image' is false only if NO image is usable at all.
"""

        retrieval_context = ""
        if retrieval_packet and retrieval_packet.get("duplicate_found"):
            retrieval_context = f"\nWARNING: RETRIEVAL MEMORY FLAGS THESE IMAGES AS REUSED FRAUD PHOTOS: {retrieval_packet.get('duplicate_details')}\n"

        user_prompt = f"""CLAIM GRAPH:
Object: {base_inputs.get('claim_object')}
Issue Type: {claim_packet.get('issue_type')}
Object Part: {claim_packet.get('object_part')}

EVIDENCE REQUIREMENTS:
{req_text}
{retrieval_context}
OBSERVATION GRAPH:
{self.format_image_evidence(image_packet)}

IMPORTANT: Only evaluate against the specific requirement above that matches the Issue Type ({claim_packet.get('issue_type')}). Ignore requirements that do not apply to this issue type.
Does the image set satisfy the applicable minimum evidence requirement above?
Return JSON only.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def _call_api():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1024
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "", 1)
            if content.endswith("```"):
                content = content[:-3]
                
            parsed = EvidenceOutput.model_validate_json(content.strip())
            return parsed.model_dump()
            
        try:
            return _call_api()
        except Exception as e:
            return {
                "object_visible": False,
                "object_mismatch": False,
                "part_visible": False,
                "damage_visible": False,
                "evidence_standard_met": False,
                "evidence_standard_met_reason": f"Error parsing LLM response: {str(e)}",
                "valid_image": False
            }
