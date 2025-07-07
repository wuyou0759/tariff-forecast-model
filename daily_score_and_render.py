#!/usr/bin/env python3
import os
import joblib
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# ——— CONFIG ———
MODEL_PATH    = "randomforestclassifier.pkl"
DATA_CSV      = "daily_data.csv"
TEMPLATE_DIR  = "templates"
TEMPLATE_NAME = "index.html.j2"
OUTPUT_HTML   = "docs/index.html"
# ——————————

def main():
    df = pd.read_csv(DATA_CSV)
  
    if 'pred' not in df.columns or df['pred'].isna().any():
        clf = joblib.load(MODEL_PATH)
        df['pred'] = df['pred'] if 'pred' in df.columns else pd.NA
        mask = df['pred'].isna()
        X_new = df.loc[mask, ['x1']]
        df.loc[mask, 'pred'] = clf.predict_proba(X_new)[:,1]
        df.to_csv(DATA_CSV, index=False)

    #Jinja
    latest  = df.iloc[-1].to_dict()
    history = df.to_dict(orient='records')

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    tpl = env.get_template(TEMPLATE_NAME)
    html = tpl.render(latest=latest, history=history)

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

if __name__ == "__main__":
    main()
