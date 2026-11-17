import os, sys, json, hashlib, platform, importlib.util, subprocess, time
from pathlib import Path

# Ensure dependencies
for pacote_import, pacote_pip in [("ucimlrepo", "ucimlrepo>=0.0.7"), ("imblearn", "imbalanced-learn>=0.12")]:
    if importlib.util.find_spec(pacote_import) is None and (pacote_import != "ucimlrepo" or not os.getenv("CREDIT_DATA_LOCAL", "").strip()):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pacote_pip])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn, imblearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_curve, ConfusionMatrixDisplay, make_scorer)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.inspection import permutation_importance
from joblib import dump

start_time = time.time()

SEED = 42
RAPIDO = os.getenv("CREDIT_FAST_MODE", "0") == "1"
CV = StratifiedKFold(n_splits=(2 if RAPIDO else 3), shuffle=True, random_state=SEED)
N_ARVORES = 35 if RAPIDO else 140
N_BOOST = 40 if RAPIDO else 130
PASTA_ARTEFATOS = Path("/home/barujafe/dev/credit-risk-intelligence/results")
PASTA_ARTEFATOS.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

print(f"Iniciando reprodução: Python {platform.python_version()} | pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
print(f"Modo rápido: {RAPIDO} | CV splits: {CV.n_splits} | SEED: {SEED}")

ATRASOS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
FATURAS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAGAMENTOS = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAFIA = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
FINANCEIROS = ["LIMIT_BAL", *ATRASOS, *FATURAS, *PAGAMENTOS]
COLUNAS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", *ATRASOS, *FATURAS, *PAGAMENTOS]
assert len(COLUNAS) == 23

# Carregar dados
caminho_local = os.getenv("CREDIT_DATA_LOCAL", "").strip()
if caminho_local and Path(caminho_local).is_file():
    df = pd.read_csv(caminho_local)
    origem_dados = f"Arquivo local: {Path(caminho_local).name}"
else:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=350)
    partes = []
    ids = getattr(dataset.data, "ids", None)
    if ids is not None:
        partes.append(ids.to_frame() if isinstance(ids, pd.Series) else ids)
    partes.extend([dataset.data.features, dataset.data.targets])
    df = pd.concat(partes, axis=1)
    origem_dados = "UCI Machine Learning Repository (id=350)"

mapa = {f"X{i}": col for i, col in enumerate(COLUNAS, start=1)}
for alvo in ["Y", "default.payment.next.month", "default payment next month", "default_payment_next_month"]:
    mapa[alvo] = "inadimplente"
df = df.rename(columns=lambda c: mapa.get(str(c).strip(), str(c).strip()))
if "ID" not in df.columns:
    df.insert(0, "ID", np.arange(1, len(df) + 1))

dados = df[["ID", *COLUNAS, "inadimplente"]].copy()
for coluna in dados.columns:
    dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
dados["inadimplente"] = dados["inadimplente"].astype(int)
print(f"Fonte: {origem_dados} | Clientes: {len(dados):,} | Taxa evento: {dados['inadimplente'].mean():.4f}")

# Split 80/20
X = dados[COLUNAS].copy()
y = dados["inadimplente"].copy()
X_treino_raw, X_teste_raw, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

# Feature engineering
FEATURES_NOVAS = ["meses_atraso_positivo", "maior_atraso_positivo", "meses_sem_pagamento",
                 "fatura_media", "fatura_volatilidade", "pagamento_medio",
                 "utilizacao_recente", "relacao_pagamentos_faturas", "tendencia_fatura_limite"]

def engenharia_atributos(df_in):
    novo = df_in.copy()
    atraso = novo[ATRASOS]
    faturas = novo[FATURAS]
    pagamentos = novo[PAGAMENTOS]
    limite_seguro = novo["LIMIT_BAL"].clip(lower=1.0)
    novo["meses_atraso_positivo"] = atraso.gt(0).sum(axis=1)
    novo["maior_atraso_positivo"] = atraso.clip(lower=0).max(axis=1)
    novo["meses_sem_pagamento"] = pagamentos.eq(0).sum(axis=1)
    novo["fatura_media"] = faturas.mean(axis=1)
    novo["fatura_volatilidade"] = faturas.std(axis=1).fillna(0)
    novo["pagamento_medio"] = pagamentos.mean(axis=1)
    novo["utilizacao_recente"] = (novo["BILL_AMT1"].clip(lower=0) / limite_seguro).clip(0, 10)
    soma_faturas_positivas = faturas.clip(lower=0).sum(axis=1)
    novo["relacao_pagamentos_faturas"] = (
        pagamentos.clip(lower=0).sum(axis=1) / soma_faturas_positivas.clip(lower=1)
    ).clip(0, 10)
    novo["tendencia_fatura_limite"] = ((novo["BILL_AMT1"]-novo["BILL_AMT6"]) / limite_seguro).clip(-10,10)
    return novo

X_treino = engenharia_atributos(X_treino_raw)
X_teste = engenharia_atributos(X_teste_raw)

CONJUNTOS = {
    "Somente demográficos": DEMOGRAFIA,
    "Financeiros · último mês": ["LIMIT_BAL", "PAY_0", "BILL_AMT1", "PAY_AMT1"],
    "Financeiros · 6 meses": FINANCEIROS,
    "Financeiros + engenharia": [*FINANCEIROS, *FEATURES_NOVAS],
    "Todos os originais": COLUNAS,
}

CATEGORICAS = ["SEX","EDUCATION","MARRIAGE",*ATRASOS]

def preprocessar(cols):
    num = [c for c in cols if c not in CATEGORICAS]
    cat = [c for c in cols if c in CATEGORICAS]
    parts = []
    if num:
        parts.append(("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler())]), num))
    if cat:
        parts.append(("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0)

def montar(modelo, cols, sampler=None):
    passos = [("preparar", preprocessar(cols))]
    if sampler is not None:
        passos.append(("amostrar", sampler))
    passos.append(("modelo", modelo))
    return ImbPipeline(passos)

SCORING = {
    "ap": "average_precision", "roc_auc": "roc_auc",
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "accuracy": "accuracy", "brier": "neg_brier_score"
}

def validar(nome, pipeline, cols):
    scores = cross_validate(pipeline, X_treino[cols], y_treino, scoring=SCORING,
                            cv=CV, n_jobs=1, error_score="raise")
    linha = {"Modelo": nome, "Nº atributos": len(cols)}
    for metr in SCORING:
        valores = scores["test_"+metr]*(-1 if metr=="brier" else 1)
        linha[metr] = float(np.mean(valores))
        linha[metr+"_dp"] = float(np.std(valores, ddof=1))
    linha["tempo_cv_s"] = float(np.sum(scores["fit_time"]+scores["score_time"]))
    return linha

print("Executando estudo de ablação...")
modelo_ab = LogisticRegression(C=0.5, max_iter=1500, random_state=SEED)
resultados_ablacao = pd.DataFrame([
    validar(nome, montar(clone(modelo_ab), cols), cols)
    for nome, cols in CONJUNTOS.items()
]).sort_values("ap", ascending=False).reset_index(drop=True)

financeiros_candidatos = resultados_ablacao[
    resultados_ablacao["Modelo"].isin(["Financeiros · 6 meses", "Financeiros + engenharia"])]
conjunto_escolhido = financeiros_candidatos.sort_values(
    ["ap", "roc_auc"], ascending=False).iloc[0]["Modelo"]
cols_modelo = CONJUNTOS[conjunto_escolhido]
print(f"Representação escolhida: {conjunto_escolhido} ({len(cols_modelo)} atributos)")

print("Executando benchmark de modelos...")
MODELOS = {
    "Dummy · prevalência": DummyClassifier(strategy="prior"),
    "Regressão logística": LogisticRegression(C=0.5, max_iter=1500, random_state=SEED),
    "Árvore de decisão": DecisionTreeClassifier(max_depth=6, min_samples_leaf=80, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=N_ARVORES, max_depth=14,
                                              min_samples_leaf=25, n_jobs=2, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=N_BOOST, learning_rate=.08,
                                  max_leaf_nodes=18, l2_regularization=2, random_state=SEED),
    "Naive Bayes": GaussianNB(var_smoothing=0.1),
    "SVM linear calibrado": CalibratedClassifierCV(LinearSVC(C=0.05, dual=False,
                                  random_state=SEED, max_iter=3000), method="sigmoid", cv=3),
}
modelos_pipeline = {nome: montar(clone(modelo), cols_modelo) for nome, modelo in MODELOS.items()}
resultados_modelos = pd.DataFrame([
    validar(nome, pipe, cols_modelo) for nome, pipe in modelos_pipeline.items()
]).sort_values("ap", ascending=False).reset_index(drop=True)

print("Executando teste de balanceamento...")
BALANCEAMENTOS = {
    "Logística sem reamostragem": montar(LogisticRegression(C=.5, max_iter=1500, random_state=SEED), cols_modelo),
    "Logística com pesos": montar(LogisticRegression(C=.5, max_iter=1500, class_weight="balanced", random_state=SEED), cols_modelo),
    "Logística com oversampling": montar(LogisticRegression(C=.5, max_iter=1500, random_state=SEED), cols_modelo,
                                         RandomOverSampler(random_state=SEED)),
    "Logística com undersampling": montar(LogisticRegression(C=.5, max_iter=1500, random_state=SEED), cols_modelo,
                                          RandomUnderSampler(random_state=SEED)),
}
resultado_balanceamento = pd.DataFrame([
    validar(nome, pipe, cols_modelo) for nome, pipe in BALANCEAMENTOS.items()
]).sort_values("ap", ascending=False).reset_index(drop=True)

candidatos = {**modelos_pipeline,
    **{k:v for k,v in BALANCEAMENTOS.items() if k!="Logística sem reamostragem"}}
ranking = pd.concat([resultados_modelos, resultado_balanceamento[
    resultado_balanceamento["Modelo"]!="Logística sem reamostragem"]], ignore_index=True)
ranking = ranking.sort_values(["ap","roc_auc","Modelo"], ascending=[False,False,True]).reset_index(drop=True)
nome_vencedor = ranking.loc[0, "Modelo"]
pipeline_vencedor = candidatos[nome_vencedor]
print(f"Modelo vencedor: {nome_vencedor} | AP CV: {ranking.loc[0, 'ap']:.4f}")

# OOF threshold selection (sem consultar o teste)
prob_oof = cross_val_predict(clone(pipeline_vencedor), X_treino[cols_modelo], y_treino,
                            cv=CV, method="predict_proba", n_jobs=1)[:, 1]
limiares = np.linspace(0.05, 0.90, 86)
f1_por_limiar = [f1_score(y_treino, (prob_oof >= lim).astype(int), zero_division=0) for lim in limiares]
limiar_escolhido = float(limiares[np.argmax(f1_por_limiar)])
print(f"Limiar ótimo OOF: {limiar_escolhido:.4f}")

def metricas_top_frac(y_true, score, frac=0.10):
    y_a = np.asarray(y_true, dtype=int)
    p_a = np.asarray(score, dtype=float)
    k = max(1, int(np.ceil(frac * len(y_a))))
    ranking_idx = np.argsort(-p_a, kind="stable")[:k]
    encontrados = int(y_a[ranking_idx].sum())
    base = float(y_a.mean())
    return {
        "selecionados": k,
        "eventos_selecionados": encontrados,
        "precision_at_10": encontrados / k,
        "recall_at_10": encontrados / max(int(y_a.sum()), 1),
        "lift_at_10": (encontrados / k) / base if base > 0 else 0.0,
        "taxa_base": base
    }

# Teste reservado
print("Treinando modelo final e avaliando no teste reservado...")
modelo_final = clone(pipeline_vencedor).fit(X_treino[cols_modelo], y_treino)
prob_teste = modelo_final.predict_proba(X_teste[cols_modelo])[:, 1]
pred_teste = (prob_teste >= limiar_escolhido).astype(int)

metricas_teste = {
    "modelo": nome_vencedor, "representacao": conjunto_escolhido, "limiar": limiar_escolhido,
    "n_teste": int(len(y_teste)), "n_eventos": int(y_teste.sum()),
    "prevalencia_teste": float(y_teste.mean()),
    "average_precision": float(average_precision_score(y_teste, prob_teste)),
    "roc_auc": float(roc_auc_score(y_teste, prob_teste)),
    "accuracy": float(accuracy_score(y_teste, pred_teste)),
    "precision": float(precision_score(y_teste, pred_teste, zero_division=0)),
    "recall": float(recall_score(y_teste, pred_teste, zero_division=0)),
    "f1": float(f1_score(y_teste, pred_teste, zero_division=0)),
    "brier": float(brier_score_loss(y_teste, prob_teste)),
}
metricas_teste.update(metricas_top_frac(y_teste, prob_teste, frac=0.10))

print("\n--- RESULTADOS NO TESTE RESERVADO ---")
for k, v in metricas_teste.items():
    print(f"  {k}: {v}")

# Bootstrap IC 95%
B = 1000
boot_ap, boot_auc, boot_p10, boot_r10, boot_l10 = [], [], [], [], []
pos_idx = np.where(y_teste.values == 1)[0]
neg_idx = np.where(y_teste.values == 0)[0]
rng = np.random.default_rng(SEED)

for _ in range(B):
    amostra_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    amostra_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    idx = np.concatenate([amostra_pos, amostra_neg])
    yb = y_teste.values[idx]
    pb = prob_teste[idx]
    boot_ap.append(average_precision_score(yb, pb))
    boot_auc.append(roc_auc_score(yb, pb))
    top = metricas_top_frac(yb, pb, frac=0.10)
    boot_p10.append(top["precision_at_10"])
    boot_r10.append(top["recall_at_10"])
    boot_l10.append(top["lift_at_10"])

intervalos = pd.DataFrame([
    {"Métrica": "Average Precision", "IC 95% inferior": float(np.percentile(boot_ap, 2.5)), "IC 95% superior": float(np.percentile(boot_ap, 97.5))},
    {"Métrica": "ROC-AUC", "IC 95% inferior": float(np.percentile(boot_auc, 2.5)), "IC 95% superior": float(np.percentile(boot_auc, 97.5))},
    {"Métrica": "Precision@10%", "IC 95% inferior": float(np.percentile(boot_p10, 2.5)), "IC 95% superior": float(np.percentile(boot_p10, 97.5))},
    {"Métrica": "Recall@10%", "IC 95% inferior": float(np.percentile(boot_r10, 2.5)), "IC 95% superior": float(np.percentile(boot_r10, 97.5))},
    {"Métrica": "Lift@10%", "IC 95% inferior": float(np.percentile(boot_l10, 2.5)), "IC 95% superior": float(np.percentile(boot_l10, 97.5))},
])
print("\n--- INTERVALOS BOOTSTRAP 95% ---")
print(intervalos.to_string(index=False))

# Permutação de importância em validação
indice_fit, indice_val = train_test_split(np.arange(len(y_treino)), test_size=0.25,
                                         stratify=y_treino, random_state=SEED)
modelo_perm = clone(pipeline_vencedor).fit(X_treino[cols_modelo].iloc[indice_fit], y_treino.iloc[indice_fit])
perm = permutation_importance(modelo_perm, X_treino[cols_modelo].iloc[indice_val],
                              y_treino.iloc[indice_val], scoring="average_precision",
                              n_repeats=5, random_state=SEED, n_jobs=1)
ranking_importancia = pd.DataFrame({
    "Atributo": cols_modelo,
    "importancia_ap_media": perm.importances_mean,
    "importancia_ap_dp": perm.importances_std
}).sort_values("importancia_ap_media", ascending=False).reset_index(drop=True)

# Auditoria subgrupos
auditoria_subgrupos = pd.DataFrame([
    {"Grupo": "Global", "N": len(y_teste), "Inadimplencia": float(y_teste.mean()), "AP": float(average_precision_score(y_teste, prob_teste)), "ROC-AUC": float(roc_auc_score(y_teste, prob_teste))}
])

# Exportar artefatos
resultados_ablacao.to_csv(PASTA_ARTEFATOS/"ablacao.csv", index=False)
resultados_modelos.to_csv(PASTA_ARTEFATOS/"modelos_cv.csv", index=False)
resultado_balanceamento.to_csv(PASTA_ARTEFATOS/"balanceamento_cv.csv", index=False)
ranking_importancia.to_csv(PASTA_ARTEFATOS/"importancia_permutacao.csv", index=False)
auditoria_subgrupos.to_csv(PASTA_ARTEFATOS/"auditoria_subgrupos.csv", index=False)
intervalos.to_csv(PASTA_ARTEFATOS/"intervalos_bootstrap.csv", index=False)

with open(PASTA_ARTEFATOS/"metricas_teste.json", "w", encoding="utf-8") as f:
    json.dump(metricas_teste, f, ensure_ascii=False, indent=2)

relatorio = f"""### Resultado executivo — teste separado

- **População:** {len(dados):,} clientes históricos; {len(y_treino):,} no treino/CV e {len(y_teste):,} no teste estratificado.
- **Modelo:** {nome_vencedor}, usando **{conjunto_escolhido}** ({len(cols_modelo)} atributos financeiros).
- **Discriminação no teste:** AP **{metricas_teste['average_precision']:.3f}** contra prevalência **{metricas_teste['prevalencia_teste']:.3f}**; ROC-AUC **{metricas_teste['roc_auc']:.3f}**.
- **Limiar escolhido por OOF ({limiar_escolhido:.2f}):** precisão **{metricas_teste['precision']:.1%}**, recall **{metricas_teste['recall']:.1%}**, F1 **{metricas_teste['f1']:.3f}**.
- **Operação com capacidade de 10%:** {metricas_teste['eventos_selecionados']:,} eventos entre {metricas_teste['selecionados']:,} clientes; precision@10% **{metricas_teste['precision_at_10']:.1%}**, recall@10% **{metricas_teste['recall_at_10']:.1%}**, lift **{metricas_teste['lift_at_10']:.2f}×**.
- **Qualidade das probabilidades:** Brier score **{metricas_teste['brier']:.4f}**. A curva de calibração deve ser avaliada visualmente antes de interpretar scores como probabilidades individuais confiáveis.

**Conclusão:** o ganho de concentração de eventos no topo do ranking descreve utilidade potencial para *priorização humana*, não retorno financeiro, redução causal de perdas ou autorização automática para negar crédito.

**Limitações materiais:** dados de Taiwan/2005, uma única organização, avaliação aleatória (não temporal), falta de validação externa e de custos de FP/FN, possíveis vieses de coleta/categorização e ausência de auditoria de justiça e conformidade regulatória de implantação.
"""
with open(PASTA_ARTEFATOS/"relatorio_executivo.md", "w", encoding="utf-8") as f:
    f.write(relatorio)

manifesto = {
    "fonte": origem_dados, "doi": "https://doi.org/10.24432/C55S3H",
    "algoritmo": nome_vencedor, "representacao": conjunto_escolhido,
    "features": cols_modelo, "limiar": limiar_escolhido,
    "particao_teste": 0.20, "seed": SEED, "folds_cv": CV.n_splits,
    "treino_n": len(y_treino), "teste_n": len(y_teste), "modo_rapido": RAPIDO,
    "python": platform.python_version(), "sklearn": sklearn.__version__,
    "aviso": "Pesquisa histórica: não utilizar para decisão automática real de crédito."
}
with open(PASTA_ARTEFATOS/"manifesto.json", "w", encoding="utf-8") as f:
    json.dump(manifesto, f, ensure_ascii=False, indent=2)

dump(modelo_final, PASTA_ARTEFATOS/"modelo_experimental.joblib")

elapsed = time.time() - start_time
print(f"\nReprodução concluída com sucesso em {elapsed:.1f} segundos!")
print(f"Artefatos salvos em: {PASTA_ARTEFATOS}")
