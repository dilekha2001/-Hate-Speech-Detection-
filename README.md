# Multi-Class Hate Speech Detection in Sinhala-English Code-Mixed Text with Emoji Integration (XLM-RoBERTa)

IT41043 Intelligent Systems, Horizon Campus. Research question: does encoding emoji as text improve macro-F1 of fine-tuned XLM-RoBERTa on three-class (neutral / offensive / hate) Facebook + YouTube comments?

Result (verified Colab T4 run, 5 paired folds): proposed **0.6786 ± 0.0291** vs baseline **0.5836 ± 0.0122** (mean gain **+0.0950**, wins **5/5 folds**, Wilcoxon **W = 0.0, p = 0.0625** — minimum possible with n = 5, so report consistency 5/5, not just p).

> All code, data, and docs live in [`Hate speech detection project/`](Hate%20speech%20detection%20project/README.md). This file is a thin wrapper — start there for full details. Paths below contain spaces, so **quote them**.

## 60-second demo (laptop, no GPU)

```bash
pip install streamlit pandas scikit-learn emoji

# From THIS (repo root) folder:
python3 -m streamlit run "Hate speech detection project/4_Presentation_and_App/app.py"
# open http://localhost:8501

# Or from inside the project folder:
cd "Hate speech detection project"
python3 -m streamlit run 4_Presentation_and_App/app.py
```

- **Tab 1 — Live Moderation Tool:** type a Sinhala-English comment with emoji (e.g. `watti amma kenek 😂`), toggle `with_emoji` (proposed: 😡 → `enraged face`) vs `no_emoji` (baseline: emoji removed), press Moderate. TF-IDF + LogisticRegression fallback (1,947 rows, CPU); uses the XLM-R checkpoint automatically if you trained with `--save_model`.
- **Tab 2 — Experimental Metrics:** live XLM-R 5-fold numbers (per-fold wins, per-class F1, confusion tables, F1 explainer), with a banner showing the source file or fallback state.
- **Tab 3 — Statistical Significance:** Wilcoxon W/p/α on the 5 fold pairs + why p = 0.0625 is the floor with n = 5.

If you hit `Error: Invalid value: File does not exist: app.py`, you ran `streamlit run app.py` from the wrong folder — use one of the two commands above.

## Repo layout

```
./
  Hate speech detection project/   ← the actual project (see its README.md for everything)
    1_Data_Preparation/            data + preprocessing.py + data_processed/{data_no_emoji,data_with_emoji,folds}.csv/json
    2_Model_Training/              train.py, dataset.py, make_folds.py, label_audit.py, collection/
    3_Evaluation_and_Stats/        evaluate.py, stats_test.py, summary_metrics.csv (live app source)
    4_Presentation_and_App/        app.py (3-tab dashboard), paper.tex, Hate_Speech_Detection.pdf
    run_full_pipeline.ipynb        Colab Cells 1–10 (train → evaluate → zip)
    requirements.txt               full deps (torch, transformers, streamlit, …)
    CHECKLIST.md                   done vs to-do
    README.md                      full documentation (start here)
  README.md                        this wrapper
```

Headline numbers live in `Hate speech detection project/3_Evaluation_and_Stats/summary_metrics.csv` (the app also accepts `outputs_final/summary/summary_metrics.csv` after a full Colab train — first hit wins).

