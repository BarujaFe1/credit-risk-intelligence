# Guia de Reprodução Independente

**Estudo:** Credit Risk Intelligence  
**Ambiente de Referência:** Linux x86_64, Python 3.14.4 / Python 3.11+, scikit-learn 1.9.1 / 1.5+, ucimlrepo 0.0.7

---

## 1. Passo a Passo de Execução

### Opção A — Execução via Script Automatizado (Recomendada)
```bash
# 1. Clonar repositório
git clone https://github.com/BarujaFe1/credit-risk-intelligence.git
cd credit-risk-intelligence

# 2. Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar pipeline completo (modo padrão: 3 folds, 1000 bootstraps)
python run_reproduction.py
```
*Tempo de execução observado:* **133.4 segundos** (Intel/AMD x86_64 moderno).

### Opção B — Execução Interativa via Jupyter Notebook
```bash
jupyter notebook Credit_Risk_Intelligence_Portfolio.ipynb
# Executar todas as células sequencialmente (Kernel -> Restart & Run All)
```

---

## 2. Comparativo Empírico: Baseline Salvo vs. Reprodução Real

| Métrica no Teste Reservado (6.000 clientes) | Baseline Salvo no Notebook | Reprodução Local (17/11/2026) | Delta Observado | Tolerância / Status |
| :--- | :---: | :---: | :---: | :---: |
| **Clientes no teste** | 6.000 | 6.000 | 0 | Exato |
| **Eventos observados (defaults)** | 1.327 | 1.327 | 0 | Exato (22,12%) |
| **Average Precision (AP)** | 0.5533 | 0.5553 | +0.0020 | Dentro da tolerância do HistGradientBoosting |
| **ROC-AUC** | 0.7803 | 0.7804 | +0.0001 | Exato |
| **Brier Score** | 0.1350 | 0.1347 | -0.0003 | Exato |
| **Clientes no Top 10%** | 600 | 600 | 0 | Exato |
| **Eventos no Top 10%** | 412 | 415 | +3 | Dentro da variação estocástica |
| **Precision@10%** | 68.67% | 69.17% | +0.50% | Consistente |
| **Recall@10%** | 31.05% | 31.27% | +0.22% | Consistente |
| **Lift@10%** | 3.10× | 3.13× | +0.03× | Consistente |
| **Bootstrap IC 95% AP** | 0.5293 – 0.5808 | 0.5303 – 0.5816 | ±0.001 | Sobreposição de 99% |
| **Bootstrap IC 95% ROC-AUC** | 0.7665 – 0.7948 | 0.7661 – 0.7952 | ±0.0005 | Sobreposição de 99% |
| **Bootstrap IC 95% Lift@10%** | 2.9390 – 3.2651 | 2.9540 – 3.2705 | ±0.01 | Sobreposição de 99% |

As diferenças mínimas decorrem de refinamentos de ponto flutuante entre versões do `scikit-learn` (1.5 vs 1.9), comprovando **alta estabilidade e reprodutibilidade numérica**.
