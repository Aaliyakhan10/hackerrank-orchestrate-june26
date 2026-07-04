import pandas as pd
from main import app

def test_pipeline():
    print("Loading sample claims...")
    claims_df = pd.read_csv("../dataset/sample_claims.csv").tail(5)
    
    for idx, row in claims_df.iterrows():
        print(f"\n{'='*60}")
        print(f"Testing Row {idx+1}: User {row['user_id']} ({row['claim_object']})")
        print(f"Claim Transcript:\n{row['user_claim']}")
        print(f"{'-'*60}")
        
        base_inputs = {
            "user_id": row["user_id"],
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"]
        }
        
        initial_state = {"base_inputs": base_inputs}
        print("Invoking LangGraph pipeline... (This calls OpenRouter APIs)")
        
        try:
            final_state = app.invoke(initial_state)
            verdict = final_state["final_verdict"]
            
            print("\n--- EXPECTED (Ground Truth) ---")
            print(f"Claim Status: {row['claim_status']}")
            print(f"Issue Type  : {row['issue_type']}")
            print(f"Object Part : {row['object_part']}")
            
            print("\n--- ACTUAL (Agent Output) ---")
            print(f"Claim Status: {verdict.get('claim_status')}")
            print(f"Issue Type  : {verdict.get('issue_type')}")
            print(f"Object Part : {verdict.get('object_part')}")
            print(f"Severity    : {verdict.get('severity')}")
            print(f"Risk Flags  : {verdict.get('risk_flags')}")
            print(f"Justification: {verdict.get('claim_status_justification')}")
            print(f"Evidence Met : {verdict.get('evidence_standard_met')}")
            
        except Exception as e:
            print(f"Error invoking graph: {str(e)}")
            
if __name__ == "__main__":
    test_pipeline()
