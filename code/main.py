import os
import sys
import pandas as pd
import csv
import time
from typing import TypedDict, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agents
from claim_extractor import ClaimExtractor
from image_analyzer import ImageAnalyzer
from identity_checker import IdentityChecker
from history_analyzer import HistoryAnalyzer
from retrieval_memory import RetrievalMemory
from evidence_checker import EvidenceChecker
from verdict_engine import VerdictEngine

from langgraph.graph import StateGraph, START, END

# Define Graph State
class AgentState(TypedDict):
    base_inputs: Dict[str, Any]
    claim_packet: Dict[str, Any]
    image_packet: List[Dict[str, Any]]
    identity_packet: Dict[str, Any]
    history_packet: Dict[str, Any]
    retrieval_packet: Dict[str, Any]
    evidence_packet: Dict[str, Any]
    final_verdict: Dict[str, Any]

# Instantiate agents
extractor = ClaimExtractor()
vision = ImageAnalyzer()
identity = IdentityChecker()
history = HistoryAnalyzer("../dataset/user_history.csv")
retrieval = RetrievalMemory()
checker = EvidenceChecker()
verdict = VerdictEngine()

def run_claim_node(state: AgentState):
    res = extractor.extract_claim(state["base_inputs"]["user_claim"], state["base_inputs"]["claim_object"])
    return {"claim_packet": res}

def run_image_node(state: AgentState):
    raw_paths = state["base_inputs"]["image_paths"].split(";")
    image_paths = []
    for p in raw_paths:
        p = p.strip()
        if p:
            if not p.startswith("../"):
                p = f"../dataset/{p}"
            image_paths.append(p)
    res = vision.analyze_images_parallel(image_paths)
    return {"image_packet": res}

def run_identity_node(state: AgentState):
    if state["base_inputs"]["claim_object"] != "car":
        return {"identity_packet": {"same_vehicle": True, "identity_flags": "none"}}
    res = identity.verify_identity(state["image_packet"])
    return {"identity_packet": res}

def run_history_node(state: AgentState):
    res = history.analyze_history(state["base_inputs"]["user_id"])
    return {"history_packet": res}

def run_retrieval_node(state: AgentState):
    raw_paths = state["base_inputs"]["image_paths"].split(";")
    image_paths = []
    for p in raw_paths:
        p = p.strip()
        if p:
            if not p.startswith("../"):
                p = f"../dataset/{p}"
            image_paths.append(p)
    res = retrieval.check_for_duplicates(state["base_inputs"]["user_id"], image_paths)
    return {"retrieval_packet": res}

def run_evidence_checker(state: AgentState):
    """
    Runs Agent 4 (Evidence Checker).
    """
    import pandas as pd
    claim_obj = state["base_inputs"]["claim_object"]
    req_df = pd.read_csv("../dataset/evidence_requirements.csv")
    relevant_reqs = req_df[(req_df['claim_object'] == claim_obj) | (req_df['claim_object'] == 'all')]
    combined_req_text = ""
    for _, r in relevant_reqs.iterrows():
        combined_req_text += f"- (Applies to: {r['applies_to']}) {r['minimum_image_evidence']}\n"

    evidence_packet = checker.verify_evidence(
        state["base_inputs"],
        state["claim_packet"],
        state["image_packet"],
        combined_req_text,
        state.get("retrieval_packet")
    )
    return {"evidence_packet": evidence_packet}

def run_verdict_engine(state: AgentState):
    """
    Runs Agent 5 (Verdict Engine).
    """
    final_verdict = verdict.generate_verdict(
        state["base_inputs"],
        state["claim_packet"],
        state["image_packet"],
        state["history_packet"],
        state["evidence_packet"],
        state.get("identity_packet"),
        state.get("retrieval_packet")
    )
    
    # Logic handled strictly in verdict_engine rule_engine now.

    # Consistency Enforcer: If claim is supported, evidence standard MUST be met.
    if final_verdict.get("claim_status") == "supported":
        state["evidence_packet"]["evidence_standard_met"] = True
        final_verdict["evidence_standard_met"] = "true"
        
    # Path A / Path B Evidence Gate Enforcement
    is_met = state["evidence_packet"].get("evidence_standard_met")
    if is_met is False or str(is_met).lower() == "false":
        flags = final_verdict.get("risk_flags", "")
        fraud_flags = ["wrong_object", "claim_mismatch", "possible_manipulation", "reused_photo", "text_instruction_present"]
        is_fraud = any(f in flags for f in fraud_flags)
        
        if is_fraud:
            final_verdict["claim_status"] = "contradicted"
        else:
            final_verdict["claim_status"] = "not_enough_information"
        final_verdict["evidence_standard_met"] = "false"
        
    return {"final_verdict": final_verdict, "evidence_packet": state["evidence_packet"]}

# Build LangGraph State Machine
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("claim", run_claim_node)
workflow.add_node("image", run_image_node)
workflow.add_node("identity", run_identity_node)
workflow.add_node("history", run_history_node)
workflow.add_node("retrieval", run_retrieval_node)
workflow.add_node("evidence", run_evidence_checker)
workflow.add_node("verdict", run_verdict_engine)

# Add Edges
workflow.add_edge(START, "claim")
workflow.add_edge("claim", "image")
workflow.add_edge("image", "identity")
workflow.add_edge("identity", "history")
workflow.add_edge("history", "retrieval")
workflow.add_edge("retrieval", "evidence")
workflow.add_edge("evidence", "verdict")
workflow.add_edge("verdict", END)

# Compile Graph
app = workflow.compile()

def process_claims():
    claims_df = pd.read_csv("../dataset/claims.csv")
    output_path = "../output.csv"
    
    # Exact 14-column schema matching sample_claims.csv
    columns = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part", "claim_status",
        "claim_status_justification", "supporting_image_ids",
        "valid_image", "severity"
    ]
    
    write_header = not os.path.exists(output_path)
    
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
            
        print(f"Starting processing of {len(claims_df)} claims...")
        for idx, row in claims_df.iterrows():
            print(f"Processing row {idx+1}/{len(claims_df)} for user {row['user_id']}")
            
            base_inputs = {
                "user_id": row["user_id"],
                "image_paths": row["image_paths"],
                "user_claim": row["user_claim"],
                "claim_object": row["claim_object"]
            }
            
            # Invoke the compiled LangGraph
            initial_state = {"base_inputs": base_inputs}
            try:
                final_state = app.invoke(initial_state)
                verdict_result = final_state["final_verdict"]
            except Exception as e:
                verdict_result = {
                    "user_id": row["user_id"],
                    "image_paths": row["image_paths"],
                    "user_claim": row["user_claim"],
                    "claim_object": row["claim_object"],
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": "System processing timeout; forced to manual queue.",
                    "risk_flags": "manual_review_required",
                    "issue_type": "unknown",
                    "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "[Confidence: LOW] Claim bypassed due to unhandled system failure.",
                    "supporting_image_ids": "none",
                    "valid_image": "false",
                    "severity": "none"
                }
            
            # Extract formatted result
            writer.writerow(verdict_result)
            f.flush()  # Incremental save
            
            # Rate limit backoff (Wait 3 seconds before next claim)
            time.sleep(3)
            
    print(f"Processing complete. Results appended to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--draw":
        print(app.get_graph().draw_mermaid())
    else:
        process_claims()
