import json, pytest
from pathlib import Path
import nbformat

REPO_ROOT = Path(__file__).parent.parent
NB_PATH = REPO_ROOT / "Credit_Risk_Intelligence_Portfolio.ipynb"
METRICS_PATH = REPO_ROOT / "results" / "metricas_teste.json"
MANIFESTO_PATH = REPO_ROOT / "results" / "manifesto.json"

def test_notebook_format_and_schema():
    assert NB_PATH.is_file(), "Notebook não encontrado"
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 35, f"Esperado ao menos 35 células, encontrado {len(nb.cells)}"

def test_no_forbidden_demographics_in_manifesto():
    assert MANIFESTO_PATH.is_file(), "manifesto.json não encontrado"
    with open(MANIFESTO_PATH, "r", encoding="utf-8") as f:
        manifesto = json.load(f)
    features = manifesto.get("features", [])
    demograficas = {"SEX", "EDUCATION", "MARRIAGE", "AGE"}
    assert not demograficas.intersection(features), "Variáveis demográficas presentes no modelo final!"
    assert len(features) == 28, f"Esperado 28 features, encontrado {len(features)}"

def test_metrics_integrity():
    assert METRICS_PATH.is_file(), "metricas_teste.json não encontrado"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["n_teste"] == 6000
    assert metrics["n_eventos"] == 1327
    assert 0.50 <= metrics["average_precision"] <= 0.60
    assert 0.75 <= metrics["roc_auc"] <= 0.82
    assert 0.65 <= metrics["precision_at_10"] <= 0.75
    assert 2.8 <= metrics["lift_at_10"] <= 3.5

def test_assets_exist():
    assets = [
        "overview.png", "pr-roc.png", "gains-top10.png",
        "confusion-matrix.png", "feature-importance.png"
    ]
    for asset in assets:
        p = REPO_ROOT / "assets" / asset
        assert p.is_file(), f"Asset {asset} ausente em assets/"
        assert p.stat().st_size > 1000, f"Asset {asset} vazio ou truncado"
