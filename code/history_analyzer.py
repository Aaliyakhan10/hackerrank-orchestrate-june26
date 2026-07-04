import pandas as pd
from typing import Dict, Any

class HistoryAnalyzer:
    def __init__(self, history_csv_path: str):
        """
        Initializes the HistoryAnalyzer by loading the user_history.csv into a pandas DataFrame.
        """
        self.history_df = pd.read_csv(history_csv_path)

    def analyze_history(self, user_id: str) -> Dict[str, Any]:
        """
        Takes structured numbers from the history dataframe and produces a deterministic risk summary.
        Uses pure arithmetic and lookup instead of an LLM.
        """
        user_rows = self.history_df[self.history_df["user_id"] == user_id]
        
        if user_rows.empty:
            # Safe fallback if user_id is not found in history
            return {
                "risk_level": "low",
                "risk_flag": None,
                "history_summary": "No prior claim history available.",
                "rejection_rate": 0.0,
                "high_frequency": False,
                "manual_review_needed": False,
            }
        
        row = user_rows.iloc[0]
        
        # Calculate totals safely to avoid divide-by-zero
        total = row["past_claim_count"] if pd.notna(row["past_claim_count"]) and row["past_claim_count"] > 0 else 1
        rejected = row["rejected_claim"] if pd.notna(row["rejected_claim"]) else 0
        
        rejection_rate = rejected / total
        
        last_90 = row["last_90_days_claim_count"] if pd.notna(row["last_90_days_claim_count"]) else 0
        high_frequency = last_90 > 3
        
        has_flags = pd.notna(row["history_flags"]) and str(row["history_flags"]).strip() != ""
        
        # Formula for risk score as provided
        risk_score = (rejection_rate * 0.6) + (0.4 if high_frequency else 0)
        
        return {
            "risk_level": "high" if risk_score > 0.4 else "medium" if risk_score > 0.2 else "low",
            "risk_flag": "user_history_risk" if risk_score > 0.4 else None,
            "history_summary": str(row["history_summary"]) if pd.notna(row["history_summary"]) else "No history summary.",
            "rejection_rate": round(rejection_rate, 2),
            "high_frequency": bool(high_frequency),
            "manual_review_needed": bool(row["manual_review_claim"] > 2) if pd.notna(row["manual_review_claim"]) else False,
        }
