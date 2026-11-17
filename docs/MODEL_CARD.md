# Model Card: Gradient Boosting para Priorização de Risco de Crédito

## 1. Visão Geral do Modelo
- **Arquitetura:** `HistGradientBoostingClassifier` (`scikit-learn`)
- **Hiperparâmetros:** `learning_rate=0.08`, `max_iter=130`, `max_leaf_nodes=18`, `l2_regularization=2.0`, `random_state=42`
- **Atributos de Entrada:** 28 preditores puramente financeiros (19 originais de faturas/pagamentos/limite + 9 features derivadas de comportamento).
- **Exclusão Proposital:** Nenhuma variável demográfica (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`) foi incluída no pipeline final.

## 2. Critério de Seleção e Validação
- **Métrica Primária:** Average Precision (AP / PR-AUC) calculada em Validação Cruzada Estratificada de 3 Folds ($K=3$).
- **Seleção do Limiar:** O limiar ótimo de decisão ($\tau = 0.26$) foi calibrado no conjunto de treino maximizando o $F_1$-score das previsões *Out-of-Fold* (OOF).

## 3. Desempenho no Teste Reservado (6.000 clientes, 1.327 eventos)
- **Average Precision:** 0.555 (Prevalência base: 0.221)
- **ROC-AUC:** 0.780
- **Brier Score:** 0.135
- **Métricas Operacionais (Top 10% da carteira):**
  - Clientes inspecionados: 600
  - Inadimplentes capturados: 415
  - Precision@10%: 69.2% (vs. 22.1% aleatório)
  - Recall@10%: 31.3%
  - Lift@10%: **3.13×**
