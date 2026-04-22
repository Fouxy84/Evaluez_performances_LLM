#!/usr/bin/env python3
"""
Rapport comparatif automatique — Baseline vs Enriched
Génère un tableau synthétique + graphiques PNG dans data/report/
Usage : python generate_report.py
"""

import json
import glob
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPORT_DIR = Path("data/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "global_score"]
METRIC_LABELS = {
    "faithfulness":      "Fidélité (Faithfulness)",
    "answer_relevancy":  "Pertinence réponse",
    "context_recall":    "Rappel contexte",
    "context_precision": "Précision contexte",
    "global_score":      "Score global",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def latest_pair():
    """Return the two most recent CSV files as (baseline_path, enriched_path).
    If only one run exists, use it for both (for solo-baseline reports)."""
    csvs = sorted(glob.glob("data/eval_results_*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError("Aucun fichier eval_results_*.csv trouvé dans data/")
    if len(csvs) == 1:
        return csvs[0], csvs[0]
    return csvs[-2], csvs[-1]


def load_pair(baseline_path, enriched_path):
    df_b = pd.read_csv(baseline_path)
    df_e = pd.read_csv(enriched_path)
    # parse contexts if stored as string
    for df in (df_b, df_e):
        if df["contexts"].dtype == object:
            import ast
            df["contexts"] = df["contexts"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df_b, df_e


def summary_table(df_b, df_e):
    rows = []
    for m in METRICS:
        b_val = df_b[m].mean() if m in df_b.columns else float("nan")
        e_val = df_e[m].mean() if m in df_e.columns else float("nan")
        delta = e_val - b_val
        pct = (delta / b_val * 100) if b_val != 0 else float("nan")
        rows.append({
            "Métrique": METRIC_LABELS.get(m, m),
            "Baseline": round(b_val, 4),
            "Enriched": round(e_val, 4),
            "Δ absolu": round(delta, 4),
            "Δ %": f"{pct:+.1f}%" if not np.isnan(pct) else "—",
        })
    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

COLORS = {"baseline": "#4C72B0", "enriched": "#DD8452"}


def plot_global_comparison(df_b, df_e):
    """Grouped bar chart — global metric comparison."""
    metrics_present = [m for m in METRICS if m in df_b.columns and m in df_e.columns]
    b_vals = [df_b[m].mean() for m in metrics_present]
    e_vals = [df_e[m].mean() for m in metrics_present]
    labels = [METRIC_LABELS.get(m, m) for m in metrics_present]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_b = ax.bar(x - width / 2, b_vals, width, label="Baseline", color=COLORS["baseline"])
    bars_e = ax.bar(x + width / 2, e_vals, width, label="Enriched", color=COLORS["enriched"])

    ax.set_title("Comparaison Baseline vs Enriched — métriques globales", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # value labels
    for bar in list(bars_b) + list(bars_e):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path = REPORT_DIR / "01_global_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_by_category(df_b, df_e):
    """Heatmap of global_score per category for baseline and enriched."""
    metric = "global_score"
    if metric not in df_b.columns or metric not in df_e.columns:
        return

    cats = sorted(set(df_b["category"].unique()) | set(df_e["category"].unique()))
    b_cat = df_b.groupby("category")[metric].mean().reindex(cats).fillna(0)
    e_cat = df_e.groupby("category")[metric].mean().reindex(cats).fillna(0)

    data = np.array([b_cat.values, e_cat.values])
    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 1.2), 3))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Baseline", "Enriched"])
    ax.set_title("Score global par catégorie", fontsize=12, pad=10)

    for i in range(2):
        for j in range(len(cats)):
            val = data[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.25 < val < 0.85 else "white")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path = REPORT_DIR / "02_score_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_delta_waterfall(df_b, df_e):
    """Delta bar chart per metric (enriched − baseline)."""
    metrics_present = [m for m in METRICS if m in df_b.columns and m in df_e.columns]
    deltas = [df_e[m].mean() - df_b[m].mean() for m in metrics_present]
    labels = [METRIC_LABELS.get(m, m) for m in metrics_present]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Δ Enriched − Baseline par métrique", fontsize=12, pad=10)
    ax.set_ylabel("Variation absolue")
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                d + (0.002 if d >= 0 else -0.006),
                f"{d:+.4f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=8)

    fig.tight_layout()
    path = REPORT_DIR / "03_delta_per_metric.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_route_distribution(df_e):
    """Pie chart of SQL vs RAG routing in enriched run."""
    if "route" not in df_e.columns:
        return
    counts = df_e["route"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    wedge_colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
           colors=wedge_colors[:len(counts)], startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Distribution du routage — mode Enriched", fontsize=11)
    fig.tight_layout()
    path = REPORT_DIR / "04_route_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_error_flags(df_b, df_e):
    """Stacked bar — error vs ok per category for baseline and enriched."""
    if "error_flag" not in df_b.columns or "error_flag" not in df_e.columns:
        return

    cats = sorted(set(df_b["category"].unique()) | set(df_e["category"].unique()))

    def flag_rate(df, cat):
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            return 0.0
        return sub["error_flag"].mean()

    b_rates = [flag_rate(df_b, c) for c in cats]
    e_rates = [flag_rate(df_e, c) for c in cats]

    x = np.arange(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, b_rates, width, label="Baseline", color=COLORS["baseline"], alpha=0.85)
    ax.bar(x + width / 2, e_rates, width, label="Enriched", color=COLORS["enriched"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.15)
    ax.set_title("Taux d'erreur (error_flag) par catégorie", fontsize=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = REPORT_DIR / "05_error_flag_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


# ── text report ───────────────────────────────────────────────────────────────

BIAS_ANALYSIS = """
══════════════════════════════════════════════════════════════════════════════
  ANALYSE CRITIQUE — BIAIS ET LIMITES DU SYSTÈME
══════════════════════════════════════════════════════════════════════════════

1. MAPPING LANGAGE NATUREL → SQL
   - Le routeur keyword-based (_is_sql_question) envoie ~toutes les questions
     en SQL (tous les mots-clés sont trop génériques : "quel", "compare"…).
   - Conséquence : les questions ambiguës ou narratives arrivent en SQL
     sans bénéfice, et les contextes RAG sont vides → context_recall = 0.
   - Amélioration suggérée : classifier avec un LLM ou un seuil de confiance.

2. GROUND TRUTH MANQUANTE
   - Aucun TestCase ne fournit de ground_truth → context_recall et
     context_precision sont structurellement nuls (RAGAS ne peut pas calculer).
   - Solution : définir des réponses de référence pour au moins les cas
     "simple" et "robustness".

3. COUVERTURE DES CAS DE TEST
   - 20 questions, 6 catégories : échantillon trop petit pour des conclusions
     statistiquement robustes (IC larges).
   - Les catégories "robustness_mixed" sont nouvelles → pas de baseline
     historique pour la comparaison.

4. BIAIS DU VECTEUR DE CONTEXTE (RAG)
   - FAISS indexe uniquement les documents chargés lors de l'indexation.
     Les questions sur des données temps-réel (5 derniers matchs, saison
     en cours) renvoient des contextes hors-sujet → hallucinations.
   - La double normalisation L2 (get_embedding appelé deux fois) est
     redondante mais inoffensive.

5. BIAIS DU MODÈLE ÉVALUATEUR
   - L'évaluateur RAGAS utilise le même modèle Mistral que le système évalué.
     Cela crée un biais de confirmation : le modèle tend à valider ses propres
     réponses (faithfulness artificiellement élevée).
   - Recommandation : utiliser un modèle tiers (GPT-4, Claude) comme juge.

6. LIMITES DU SCORE GLOBAL
   - global_score = moyenne(faithfulness, answer_relevancy) ignore
     context_recall et context_precision → masque les défauts de retrieval.
"""


def write_text_report(table_df, baseline_path, enriched_path):
    report_path = REPORT_DIR / "rapport_comparatif.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("══════════════════════════════════════════════════════════════════\n")
        f.write("  RAPPORT COMPARATIF — ÉVALUATION RAG AVEC RAGAS\n")
        f.write(f"  Baseline : {Path(baseline_path).name}\n")
        f.write(f"  Enriched : {Path(enriched_path).name}\n")
        f.write("══════════════════════════════════════════════════════════════════\n\n")
        f.write("TABLEAU SYNTHÉTIQUE\n")
        f.write(table_df.to_string(index=False))
        f.write("\n")
        f.write(BIAS_ANALYSIS)
        f.write("\nGraphiques générés dans : data/report/\n")
    print(f"  ✓ {report_path}")
    return report_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Générer le rapport comparatif baseline vs enriched")
    parser.add_argument("--baseline", default=None, help="Chemin vers le CSV baseline")
    parser.add_argument("--enriched", default=None, help="Chemin vers le CSV enriched")
    args = parser.parse_args()

    if args.baseline and args.enriched:
        baseline_path, enriched_path = args.baseline, args.enriched
    else:
        baseline_path, enriched_path = latest_pair()

    print(f"Baseline : {baseline_path}")
    print(f"Enriched : {enriched_path}\n")

    df_b, df_e = load_pair(baseline_path, enriched_path)

    table = summary_table(df_b, df_e)

    print("─── Tableau synthétique ───────────────────────────────────────────")
    print(table.to_string(index=False))
    print()

    print("─── Génération des graphiques ─────────────────────────────────────")
    plot_global_comparison(df_b, df_e)
    plot_by_category(df_b, df_e)
    plot_delta_waterfall(df_b, df_e)
    plot_route_distribution(df_e)
    plot_error_flags(df_b, df_e)

    print()
    print("─── Rapport texte ─────────────────────────────────────────────────")
    write_text_report(table, baseline_path, enriched_path)

    print()
    print("─── Analyse critique ──────────────────────────────────────────────")
    print(BIAS_ANALYSIS)


if __name__ == "__main__":
    main()
