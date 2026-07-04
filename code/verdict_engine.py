import os
import json
import base64
from typing import Dict, Any, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

class VerdictOutput(BaseModel):
    contradiction_found: bool
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    severity: str
    risk_flags_generated: List[str]

class VerdictEngine:
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

    def format_image_evidence(self, image_packet: List[Dict[str, Any]]) -> str:
        image_summaries = ""
        for i, img in enumerate(image_packet):
            path = img.get('image_path', f'img_{i}')
            img_id = os.path.splitext(os.path.basename(path))[0]
            image_summaries += f"Image ID: {img_id}\n"
            image_summaries += f"  - Parts Visible: {img.get('parts_visible')}\n"
            image_summaries += f"  - Damage Detected: {json.dumps(img.get('damage_detected', []))}\n"
            image_summaries += f"  - Quality Flags: {img.get('quality_flags')}\n\n"
        return image_summaries

    def generate_verdict(self, base_inputs, claim_packet, image_packet, history_packet, evidence_packet, identity_packet=None, retrieval_packet=None):
        import os
        # --- DETERMINISTIC RULE ENGINE ---
        status = "not_enough_information"
        confidence = "low"
        severity = "none"
        risk_flags = []
        
        # 1. Identity Gate: Wrong Object / Mismatch
        is_wrong_object = False
        if identity_packet and identity_packet.get("same_vehicle") is False:
            is_wrong_object = True
        elif evidence_packet.get("object_mismatch") and not evidence_packet.get("part_visible"):
            is_wrong_object = True
            
        if is_wrong_object:
            status = "contradicted"
            confidence = "high"
            risk_flags.append("wrong_object")
        # 2. Object Visibility Gate
        elif not evidence_packet.get("object_visible"):
            status = "not_enough_information"
            confidence = "low"
        # 3. Part Visibility Gate
        elif not evidence_packet.get("part_visible"):
            # If the part is not visible, we cannot confidently support or contradict.
            status = "not_enough_information"
            confidence = "low"
        # 4. Damage Gate
        elif claim_packet.get("issue_type") not in ["unknown", "none"]:
            
            # Adversarial Text Gate
            text_instruction = False
            for img in image_packet:
                txt = img.get("text_detected", "").lower()
                if "approve this claim" in txt or "tamper evident" in txt or "instruction" in txt:
                    text_instruction = True
                    break
                    
            if text_instruction:
                risk_flags.append("text_instruction_present")
                risk_flags.append("manual_review_required")
                # Removed 'status = "contradicted"' to prevent blind rejection
                
            elif evidence_packet.get("damage_visible"):
                # Damage is visible!
                highest_conf = 0.0
                for img in image_packet:
                    for d in img.get("damage_detected", []):
                        if d.get("confidence", 0.0) > highest_conf:
                            highest_conf = d.get("confidence", 0.0)
                            severity = d.get("severity", "low")
                if highest_conf > 0 and highest_conf < 0.5:
                    status = "not_enough_information"
                    confidence = "low"
                else:
                    status = "supported"
                    confidence = "high"
            else:
                if claim_packet.get("issue_type") == "missing_part":
                    status = "not_enough_information"
                    confidence = "low"
                elif evidence_packet.get("object_visible") and evidence_packet.get("part_visible"):
                    risk_flags.append("damage_not_visible")
                    status = "contradicted"
                    confidence = "high"
                else:
                    status = "not_enough_information"
                    confidence = "low"
        else:
            # No specific issue claimed
            if claim_packet.get("object_part") not in ["unknown", "none"] and not evidence_packet.get("part_visible"):
                status = "not_enough_information"
                confidence = "low"
            else:
                status = "supported"
                confidence = "medium"
            
        # Merge risk flags
        if history_packet and history_packet.get("risk_flag"):
            hf = history_packet.get("risk_flag").split(";")
            for f in hf:
                if f and f != "none" and f not in risk_flags:
                    risk_flags.append(f)
        if history_packet and history_packet.get("manual_review_needed"):
            if "manual_review_required" not in risk_flags:
                risk_flags.append("manual_review_required")
                
        for img in image_packet:
            for f in img.get("quality_flags", []):
                if f and f.lower() != "none" and f != "error" and f not in risk_flags:
                    risk_flags.append(f)
                    
        if retrieval_packet and retrieval_packet.get("duplicate_found"):
            risk_flags.append("reused_photo")
            status = "contradicted"

        risk_flags_str = "none" if not risk_flags else ";".join(sorted(risk_flags))
        
        # Determine valid_image overrides
        passive_flags = ["blurry_image", "wrong_angle"]
        if any(f in risk_flags for f in passive_flags):
            evidence_packet["valid_image"] = False

        # Gather supporting images
        supp_ids = []
        for img in image_packet:
            if "blurry_image" not in img.get("quality_flags", []):
                path = img.get("image_path", "")
                if path:
                    name = os.path.splitext(os.path.basename(path))[0]
                    supp_ids.append(name)
                    
        # Structural Grounding: If status is supported, ensure we don't output "none"
        if status == "supported" and not supp_ids:
            for img in image_packet:
                path = img.get("image_path", "")
                if path:
                    name = os.path.splitext(os.path.basename(path))[0]
                    supp_ids.append(name)
                    
        supp_str = ";".join(supp_ids) if supp_ids else "none"

        # --- LLM JUSTIFICATION GENERATOR ---
        system_prompt = f"""You are the Final Verdict Explainer.
Your job is to generate ONLY a natural language justification for the claim decision that has ALREADY BEEN MADE.

The deterministic rule engine has decided:
Claim Status: {status}
Risk Flags: {risk_flags_str}

Write a 1-2 sentence justification explaining this decision. 
If status is 'supported', explain how the image shows the claimed damage.
If status is 'contradicted', explain why the image contradicts the claim (e.g. wrong object, or damage absent).
If status is 'not_enough_information', explain what is missing (e.g. part not visible, image too zoomed in). CRITICAL: Do NOT use strong words like "contradicts" or "contradicted" when the status is not_enough_information. Keep a neutral tone and state that the evidence is insufficient.

Return a valid JSON object:
{{
  "claim_status_justification": "your 1-2 sentence explanation"
}}
"""
        observations = self.format_image_evidence(image_packet)
        user_text = f"CLAIM:\nObject: {base_inputs.get('claim_object')}\nPart: {claim_packet.get('object_part')}\nIssue: {claim_packet.get('issue_type')}\n\nOBSERVATIONS:\n{observations}\n\nEVIDENCE MET: {evidence_packet.get('evidence_standard_met')}\nREASON: {evidence_packet.get('evidence_standard_met_reason')}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ]

        # Attach images for visual context
        for img in image_packet:
            path = img.get('image_path')
            if path and os.path.exists(path):
                try:
                    b64 = self._encode_image(path)
                    mime_type = "image/jpeg"
                    if path.lower().endswith(".png"):
                        mime_type = "image/png"
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"}
                    })
                except Exception:
                    pass

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
                
            parsed = json.loads(content.strip())
            return {
                "user_id": base_inputs.get("user_id"),
                "image_paths": base_inputs.get("image_paths"),
                "user_claim": base_inputs.get("user_claim"),
                "claim_object": base_inputs.get("claim_object"),
                "evidence_standard_met": str(evidence_packet.get("evidence_standard_met", "false")).lower(),
                "evidence_standard_met_reason": evidence_packet.get("evidence_standard_met_reason", ""),
                "risk_flags": risk_flags_str,
                "claim_status": status,
                "issue_type": claim_packet.get("issue_type", "unknown"),
                "object_part": claim_packet.get("object_part", "unknown"),
                "severity": severity,
                "claim_status_justification": f"[Confidence: {confidence.upper()}] " + parsed.get("claim_status_justification", "Decision based on deterministic rules."),
                "supporting_image_ids": supp_str,
                "valid_image": str(evidence_packet.get("valid_image", "false")).lower()
            }
            
        try:
            return _call_api()
        except Exception as e:
            return {
                "user_id": base_inputs.get("user_id"),
                "image_paths": base_inputs.get("image_paths"),
                "user_claim": base_inputs.get("user_claim"),
                "claim_object": base_inputs.get("claim_object"),
                "evidence_standard_met": str(evidence_packet.get("evidence_standard_met", "false")).lower(),
                "evidence_standard_met_reason": evidence_packet.get("evidence_standard_met_reason", ""),
                "risk_flags": risk_flags_str,
                "claim_status": status,
                "issue_type": claim_packet.get("issue_type", "unknown"),
                "object_part": claim_packet.get("object_part", "unknown"),
                "severity": severity,
                "claim_status_justification": f"[Confidence: {confidence.upper()}] Decision based on deterministic rules. LLM formatting error: {str(e)}",
                "supporting_image_ids": supp_str,
                "valid_image": str(evidence_packet.get("valid_image", "false")).lower()
            }
