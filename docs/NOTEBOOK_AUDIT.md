# Auditoria Metodológica e de Integridade do Notebook

**Estudo:** Credit Risk Intelligence  
**Dataset:** UCI Default of Credit Card Clients (Taiwan, 2005)  
**Data da Auditoria:** 17 de novembro de 2026  
**Auditor:** Antigravity Autonomous Agent (em nome de Felipe Baruja)

---

## 1. Matriz de Auditoria (Achado / Risco / Solução / Evidência / Situação)

| # | Achado | Risco | Solução Adotada | Evidência | Situação |
|---|---|---|---|---|---|
| **01** | `ID` e `inadimplente` presentes no dataset original. | Vazamento direto do alvo ou sobreajuste a IDs sequenciais. | Remoção explícita de `ID` e `inadimplente` da matriz $X$ antes da partição. | `assert 'inadimplente' not in COLUNAS and 'ID' not in COLUNAS` | **RESOLVIDO** |
| **02** | Partição de treino e teste. | Vazamento entre particionamento ou desbalanceamento de classes no teste. | `train_test_split(..., test_size=0.20, stratify=y, random_state=42)` garantindo 24.000 treino / 6.000 teste e índices disjuntos. | `assert set(X_treino_raw.index).isdisjoint(set(X_teste_raw.index))` | **RESOLVIDO** |
| **03** | Engenharia de atributos (`engenharia_atributos`). | Vazamento de estatísticas globais (ex: média do teste no treino). | A função é puramente intra-linha (opera por cliente: soma de atrasos, média de faturas do próprio cliente, razão fatura/limite). Aplicada separadamente. | `X_treino = engenharia_atributos(X_treino_raw)`, `X_teste = engenharia_atributos(X_teste_raw)` | **RESOLVIDO** |
| **04** | Imputação e escalonamento numérico/categórico. | Vazamento de média/mediana do conjunto de teste nos transformadores. | Uso estrito de `ColumnTransformer` dentro de pipelines da `scikit-learn` / `imblearn`. O `fit` ocorre exclusivamente dentro dos folds da CV e no conjunto de treino final. | `ImbPipeline([("preparar", preprocessar(cols)), ("modelo", modelo)])` | **RESOLVIDO** |
| **05** | Variáveis demográficas (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`). | Risco ético, viés de grupos e conformidade em modelos de risco. | Estudo de ablação comparou subconjuntos. O modelo vencedor selecionado (`Financeiros + engenharia`) **exclui** totalmente os preditores demográficos. | `assert not set(cols_modelo).intersection(["SEX","EDUCATION","MARRIAGE","AGE"])` | **RESOLVIDO** |
| **06** | `UndefinedMetricWarning` em precision/F1 quando modelos não previam positivos ao limiar de 0.50. | Warnings no log e possível comportamento indefinido em métricas pontuais. | Configuração de `make_scorer(precision_score, zero_division=0)` e `make_scorer(f1_score, zero_division=0)`. | Logs de CV sem warnings; cálculo determinístico. | **RESOLVIDO** |
| **07** | Escolha do limiar de classificação. | Uso do teste para afinar hiperparâmetros ou limiares (vazamento). | O limiar de 0.26 foi escolhido estritamente via previsões *Out-of-Fold* (OOF) do conjunto de treino via `cross_val_predict`. O teste só foi tocado após a fixação do limiar. | Código na célula 26 sem menção a `X_teste` ou `y_teste`. | **RESOLVIDO** |
| **08** | Incerteza estatística no teste. | Afirmação pontual sem margem de erro ou intervalo de confiança. | Aplicação de bootstrap estratificado ($B=1.000$) no teste para AP, ROC-AUC, Precision@10% e Lift@10%. | Tabela de IC 95% gerada e exportada em `results/intervalos_bootstrap.csv`. | **RESOLVIDO** |
| **09** | Comunicação de resultados. | Exagerar utilidade afirmando "redução de inadimplência" ou "sistema bancário em produção". | Linguagem estritamente readequada: priorização de capacidade operacional (top 10% da carteira para mesa humana), dados de Taiwan 2005 e estudo acadêmico. | Documentação e README alinhados. | **RESOLVIDO** |

---

## 2. Verificação de PII e Dados Sensíveis
- O dataset UCI contém clientes anônimos de 2005 sem nomes, números de documento, endereços ou dados bancários reais.
- Nenhum dado pessoal, token de API, chave privada ou credencial foi inserido no repositório.
