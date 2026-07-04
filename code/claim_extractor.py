import os
import json
from openai import OpenAI
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, ValidationError

class ClaimOutput(BaseModel):
    extracted_claim_summary: str
    issue_type: str
    object_part: str

class ClaimExtractor:
    def __init__(self, model_name="google/gemma-4-31b-it:free"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model_name = model_name

    def extract_claim(self, user_claim_transcript: str, claim_object: str) -> Dict[str, Any]:
        """
        Extracts the user claim, issue type, and object part from the chat transcript.
        """
        
        system_prompt = f"""You are a data extraction agent for a damage claim processing system.
Your job is to read a chat transcript between a Customer and Support, and extract exactly three things:
1. 'extracted_claim_summary': A very concise 1-2 sentence summary of the specific physical damage claim the user wants reviewed.
2. 'issue_type': The type of issue. Allowed values: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown.
3. 'object_part': The part of the object that is damaged. 
   - If claim_object is 'car', allowed values: front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, unknown.
   - If claim_object is 'laptop', allowed values: screen, keyboard, trackpad, hinge, lid, corner, port, base, body, unknown.
   - If claim_object is 'package', allowed values: box, package_corner, package_side, seal, label, contents, item, unknown.
   
The object type is: {claim_object}

If the user describes multiple distinct issues (e.g., door dent AND bumper damage), extract ONLY the primary or most severe physical issue. Do not return "unknown" just because there are multiple.

Return ONLY a valid JSON object with these exact keys: "extracted_claim_summary", "issue_type", "object_part". No markdown formatting outside of the JSON.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_claim_transcript}
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
            
            # Pydantic validation (will raise exception on bad schema, triggering retry)
            parsed = ClaimOutput.model_validate_json(content.strip())
            return parsed.model_dump()
            
        try:
            return _call_api()
        except Exception as e:
            return {
                "extracted_claim_summary": "Error parsing LLM response",
                "issue_type": "unknown",
                "object_part": "unknown",
                "raw_response": str(e)
            }
