import os
import pandas as pd


def _read_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()


def load_project_data(base_dir):
    scored_path = os.path.join(base_dir, "data", "credit_scored_data.csv")
    feature_path = os.path.join(base_dir, "data", "data_feature.csv")
    p1_path = os.path.join(base_dir, "reports", "person1_summary_report.csv")
    p2_path = os.path.join(base_dir, "reports", "person2_summary_report.csv")

    return {
        "scored_df": _read_csv(scored_path),
        "feature_df": _read_csv(feature_path),
        "summary_p1": _read_csv(p1_path),
        "summary_p2": _read_csv(p2_path),
    }


def load_reports(base_dir):
    return {
        "model_comparison": _read_csv(os.path.join(base_dir, "reports", "model_comparison_table.csv")),
        "feature_importance": _read_csv(os.path.join(base_dir, "reports", "feature_importance.csv")),
        "shap_importance": _read_csv(os.path.join(base_dir, "reports", "shap_feature_importance.csv")),
        "threshold": _read_csv(os.path.join(base_dir, "reports", "threshold_optimization.csv")),
    }


def safe_read_image(base_dir, relative_path):
    path = os.path.join(base_dir, relative_path)
    return path if os.path.exists(path) else None
