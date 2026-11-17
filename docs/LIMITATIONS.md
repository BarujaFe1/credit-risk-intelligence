# Limitações Éticas, Metodológicas e Técnicas

Este documento formaliza as restrições e limites de aplicação do presente estudo:

1. **Uso Estritamente Acadêmico e Investigativo:**
   Este trabalho é um estudo empírico de Ciência de Dados e Estatística conduzido no contexto acadêmico do ICMC-USP. **Não constitui e não deve ser utilizado como motor de concessão, negação ou precificação de crédito no sistema financeiro real.**

2. **Ausência de Matriz de Custos Financeiros:**
   A otimização de limiar por $F_1$-score é um exercício metodológico padrão. Na prática bancária, decisões de crédito exigem balanceamento explícito entre custo de perda (Falso Negativo) e custo de oportunidade (Falso Positivo), dependendo do spread, taxa de juros e funding.

3. **Validação Não Temporal:**
   O particionamento 80/20 utilizado foi aleatório estratificado por indivíduo. Modelos em produção devem ser validados em esteiras *Out-of-Time* (OOT) para medir degradação temporal e drift econômico.

4. **Incerteza Amostral:**
   Os intervalos de confiança obtidos por bootstrap refletem a variabilidade da amostra de teste de Taiwan de 2005 sob o modelo treinado. Não garantem estabilidade futura em outras populações.
