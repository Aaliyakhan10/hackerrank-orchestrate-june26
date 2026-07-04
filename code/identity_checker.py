import os
import json
from typing import Dict, Any, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

class IdentityOutput(BaseModel):
    same_vehicle: bool
    identity_flags: str

class IdentityChecker:
    def __init__(self, model_name="google/gemma-4-31b-it:free"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model_name = model_name

    def verify_identity(self, image_packet: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cross-references multiple observations to ensure they are of the same object.
        """
        # If there's only 1 image or none, there's no mismatch risk between images
        if len(image_packet) <= 1:
            return {"same_vehicle": True, "identity_flags": "none"}
            
        system_prompt = """You are an Identity Verification Agent.
You will be given multiple Image Observations. Your job is to verify if all images show the SAME object/vehicle.
Compare 'vehicle_color_or_features', 'object_class', and 'text_detected' across images.

Return ONLY a valid JSON object:
{
  "same_vehicle": true | false,
  "identity_flags": "none" | "claim_mismatch" | "wrong_object"
}
"""
        observations = ""
        for i, img in enumerate(image_packet):
            path = img.get('image_path', f'Unknown_{i}')
            observations += f"Image {i+1} ({path}):\n"
            observations += f"- Object Class: {img.get('object_class')}\n"
            observations += f"- Color/Features: {img.get('vehicle_color_or_features')}\n"
            observations += f"- Text/Plates: {img.get('text_detected')}\n\n"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Compare these observations and output identity verification:\n{observations}"}
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
                
            parsed = IdentityOutput.model_validate_json(content.strip())
            return parsed.model_dump()
            
        try:
            return _call_api()
        except Exception as e:
            # Fallback to true if API fails so we don't erroneously block valid claims
            return {
                "same_vehicle": True, 
                "identity_flags": f"error: {str(e)}"
            }
