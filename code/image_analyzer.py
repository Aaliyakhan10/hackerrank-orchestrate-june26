import os
import json
import base64
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

class DamageObservation(BaseModel):
    part: str
    type: str
    severity: str
    confidence: float
    bounding_box: List[float] = Field(default_factory=list)

class ImageObservation(BaseModel):
    object_class: str
    vehicle_color_or_features: str
    text_detected: str
    parts_visible: List[str]
    damage_detected: List[DamageObservation]
    quality_flags: List[str]

class ImageAnalyzer:
    def __init__(self, model_name="google/gemma-4-31b-it:free"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model_name = model_name

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyzes a single image and returns visible_damage, quality_flags, confidence, and relevant_part.
        """
        # Ensure image path is correct or exists
        if not os.path.exists(image_path):
            return {
                "image_path": image_path,
                "object_class": "unknown",
                "vehicle_color_or_features": "unknown",
                "text_detected": "none",
                "parts_visible": [],
                "damage_detected": [],
                "quality_flags": ["error: file not found"],
                "error": True
            }

        try:
            base64_image = self._encode_image(image_path)
            
            # Determine mime type based on extension
            mime_type = "image/jpeg"
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith(".webp"):
                mime_type = "image/webp"

            system_prompt = (
                "You are an Observation Extraction Agent for a damage claim system. "
                "Key discipline: List ONLY observable physical facts. Do not infer intent or cause. "
                "You must return a JSON object with exactly these keys: "
                "1. 'object_class': 'car', 'laptop', 'package', etc. "
                "2. 'vehicle_color_or_features': The color and visual make/model/features of the object. "
                "3. 'text_detected': Any license plates, text, logos, or labels visible. If none, 'none'. "
                "4. 'parts_visible': A list of all object parts clearly visible. "
                "5. 'damage_detected': A list of dictionaries, each containing 'part', 'type' (e.g. 'dent', 'scratch'), 'severity', 'confidence' (0.0 to 1.0), and 'bounding_box'. Severity MUST be 'low' for scratches, light stains, or small cosmetic marks; 'medium' for standard dents or tears; and 'high' for shattered glass, missing parts, or major structural failure. ONLY detect damage if visually obvious and undeniable. Avoid hallucinating false positives like minor packaging folds as tears. If unsure, omit it. "
                "6. 'quality_flags': A list of strings for quality issues like 'blurry_image', 'wrong_angle', or 'none'. "
                "Return ONLY valid JSON."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image according to the system prompt."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
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
                
                parsed = ImageObservation.model_validate_json(content.strip())
                return parsed.model_dump()
                
            parsed = _call_api()
            parsed["image_path"] = image_path
            parsed["error"] = False
            return parsed

        except Exception as e:
            return {
                "image_path": image_path,
                "object_class": "unknown",
                "vehicle_color_or_features": "unknown",
                "text_detected": "none",
                "parts_visible": [],
                "damage_detected": [],
                "quality_flags": [f"error: {str(e)}"],
                "error": True
            }

    def analyze_images_parallel(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Runs analyze_single_image in parallel across all provided image paths.
        """
        results = []
        # Run in parallel, capping workers to 10 or number of images
        max_workers = min(len(image_paths), 10) if image_paths else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self.analyze_single_image, path): path for path in image_paths}
            for future in as_completed(future_to_path):
                results.append(future.result())
        return results
