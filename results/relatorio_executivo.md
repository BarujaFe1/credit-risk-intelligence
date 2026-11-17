### Resultado executivo — teste separado

- **População:** 30,000 clientes históricos; 24,000 no treino/CV e 6,000 no teste estratificado.
- **Modelo:** Gradient Boosting, usando **Financeiros + engenharia** (28 atributos financeiros).
- **Discriminação no teste:** AP **0.555** contra prevalência **0.221**; ROC-AUC **0.780**.
- **Limiar escolhido por OOF (0.26):** precisão **49.6%**, recall **59.2%**, F1 **0.540**.
- **Operação com capacidade de 10%:** 415 eventos entre 600 clientes; precision@10% **69.2%**, recall@10% **31.3%**, lift **3.13×**.
- **Qualidade das probabilidades:** Brier score **0.1347**. A curva de calibração deve ser avaliada visualmente antes de interpretar scores como probabilidades individuais confiáveis.

**Conclusão:** o ganho de concentração de eventos no topo do ranking descreve utilidade potencial para *priorização humana*, não retorno financeiro, redução causal de perdas ou autorização automática para negar crédito.

**Limitações materiais:** dados de Taiwan/2005, uma única organização, avaliação aleatória (não temporal), falta de validação externa e de custos de FP/FN, possíveis vieses de coleta/categorização e ausência de auditoria de justiça e conformidade regulatória de implantação.
