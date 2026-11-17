# Data Card: Default of Credit Card Clients (Taiwan, 2005)

## 1. Resumo do Dataset
- **Nome:** Default of Credit Card Clients Dataset
- **Origem:** Chung Hua University e Universidade de Tamkang, Taiwan (2005)
- **Distribuição:** UCI Machine Learning Repository (ID: 350)
- **DOI:** https://doi.org/10.24432/C55S3H
- **Licença:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## 2. Características Amostrais
- **Tamanho total:** 30.000 observações (linhas)
- **Variável alvo:** `default payment next month` (binária: 1 = inadimplente, 0 = adimplente)
- **Taxa de inadimplência (prevalência):** 22,12% (6.636 inadimplentes / 23.364 adimplentes)
- **Período temporal coberto:** Abril de 2005 a Setembro de 2005 (6 meses de histórico)

## 3. Descrição dos Atributos Originais
1. `LIMIT_BAL`: Montante do crédito concedido em NT Dollars (inclui crédito individual e familiar).
2. `SEX`: Gênero (1 = masculino, 2 = feminino) — *Excluído do modelo final*.
3. `EDUCATION`: Escolaridade (1 = pós-graduação, 2 = universidade, 3 = ensino médio, 4 = outros) — *Excluído do modelo final*.
4. `MARRIAGE`: Estado civil (1 = casado, 2 = solteiro, 3 = outros) — *Excluído do modelo final*.
5. `AGE`: Idade em anos — *Excluído do modelo final*.
6. `PAY_0` a `PAY_6`: Histórico de atrasos de pagamento nos últimos 6 meses (-1 = pago pontualmente, -2 = sem consumo, 0 = uso de crédito rotativo, 1 a 8 = meses de atraso).
7. `BILL_AMT1` a `BILL_AMT6`: Valor da fatura mensal em NT Dollars (Setembro a Abril de 2005).
8. `PAY_AMT1` a `PAY_AMT6`: Valor efetivamente pago na fatura anterior em NT Dollars.

## 4. Limitações de Domínio
- **Geográfica/Temporal:** Base coletada exclusivamente em Taiwan em 2005. Não reflete a macroeconomia, juros ou hábitos de crédito do Brasil em 2026.
- **Divisão Aleatória vs. Fora do Tempo:** O split foi estratificado por cliente, não temporal (out-of-time), devido à estrutura agregada do arquivo UCI.
