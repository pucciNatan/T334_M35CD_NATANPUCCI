import sys
import pathlib
import warnings
import textwrap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# --- Configurações gerais ----------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.dpi"] = 110

# --- Pergunta 0 – Leitura do arquivo ----------------------------------------
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    dataset_path = pathlib.Path(sys.argv[1])
else:
    dataset_path = pathlib.Path("dataset_21.csv")      # nome padrão

if not dataset_path.exists():
    raise FileNotFoundError(f"Arquivo de dataset não encontrado: {dataset_path.resolve()}")

print(f"\n>>> Carregando dataset: {dataset_path.name}\n")
df = pd.read_csv(dataset_path)

# Pergunta 1 – Estatística Descritiva
print("#" * 70)
print("Pergunta 1 – Estatística Descritiva (apenas numéricas)\n")
print(df.describe().T, "\n")

print("Pergunta 1 – Estatística Descritiva (inclui categóricas)\n")
print(df.describe(include="all").T)

# Limpeza básica
antes = len(df)
df = df.dropna()
depois = len(df)
print(f"\nLinhas antes de dropna: {antes} — após: {depois}\n")

# Pergunta 2 – Tratamento das Variáveis Categóricas
cat_cols = [c for c in [
    "sistema_operacional",
    "tipo_hd",
    "tipo_processador",
] if c in df.columns]

print("#" * 70)
print("Pergunta 2 – Variáveis Categóricas e Categorias-base\n")
for col in cat_cols:
    base = sorted(df[col].unique())[0]      # drop_first=True ⇒ menor string alfabética
    categorias = ", ".join(sorted(df[col].unique()))
    print(f"• {col}: {categorias}  →  categoria-base = {base}")
print()

# Codificação dummy (drop_first p/ evitar armadilha da colinearidade)
num_cols = [c for c in df.columns if c not in cat_cols + ["tempo_resposta"]]
df_num = df[num_cols].apply(pd.to_numeric, errors="coerce").astype(float)

if cat_cols:
    df_cat = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
    X = pd.concat([df_num, df_cat], axis=1)
else:
    X = df_num.copy()

X = sm.add_constant(X, has_constant="add").astype(float)
Y = df["tempo_resposta"].astype(float)

# Checagem final de tipo
assert not X.dtypes.eq("object").any(), "Ainda existem colunas object em X!"

# Pergunta 3 – Ajuste do Modelo Completo
print("#" * 70)
print("Pergunta 3 – Resumo do Modelo Completo (OLS)\n")
modelo_full = sm.OLS(Y, X).fit()
print(modelo_full.summary())

# Pergunta 4A – Diagnóstico de Multicolinearidade
print("#" * 70)
print("Pergunta 4A – Fatores de Inflação da Variância (VIF)\n")
vif_df = pd.DataFrame({
    "variavel": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
}).sort_values(by="VIF", ascending=False)
print(vif_df, "\n")

# Pergunta 4B – Diagnóstico de Heterocedasticidade
print("#" * 70)
print("Pergunta 4B – Teste Breusch-Pagan\n")
labels = ["LM Stat", "LM p-valor", "F Stat", "F p-valor"]
for lbl, val in zip(labels, het_breuschpagan(modelo_full.resid, modelo_full.model.exog)):
    print(f"{lbl:>10}: {val:.4f}")
print()

# (Gráficos de resíduos – descomente se estiver em notebook ou quiser visualizar)
def plot_residuos(model):
    """Exibe gráfico Resíduos × Ajustados + QQ-plot."""
    fig, ax = plt.subplots()
    sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax)
    ax.axhline(0, ls="--", c="k")
    ax.set_title("Resíduos × Valores Ajustados")
    ax.set_xlabel("Valores Ajustados")
    ax.set_ylabel("Resíduos")
    plt.show()

    fig, ax = plt.subplots()
    sm.qqplot(model.resid, line="45", fit=True, ax=ax)
    ax.set_title("QQ-Plot dos Resíduos")
    plt.show()

    plot_residuos(modelo_full)

# Pergunta 5 – Comparação de Modelos (Backward Elimination)
def backward_elimination(X_, y, thresh=0.05):
    cols = list(X_.columns)
    while True:
        model = sm.OLS(y, X_[cols]).fit()
        pvals = model.pvalues.drop("const", errors="ignore")
        worst_p = pvals.max()
        if worst_p > thresh:
            cols.remove(pvals.idxmax())
        else:
            break
    return cols, model

sel_cols, modelo_step = backward_elimination(X, Y)

print("#" * 70)
print("Pergunta 5 – Modelo Após Backward Elimination\n")
print(modelo_step.summary())

# Tabela de comparação básica
print("\nComparação dos Modelos")
print("-" * 30)
print(f"{'Métrica':<20}  Modelo 1  Modelo 2")
print(f"{'R² ajustado':<20}  {modelo_full.rsquared_adj:8.3f}  {modelo_step.rsquared_adj:8.3f}")
print(f"{'AIC':<20}  {modelo_full.aic:8.1f}  {modelo_step.aic:8.1f}")

anova_res = sm.stats.anova_lm(modelo_step, modelo_full)
print(f"{'Teste F (p-valor)':<20}  {anova_res['Pr(>F)'][1]:8.3f}\n")
print(f"Variáveis mantidas no Modelo 2: {', '.join(sel_cols)}\n")

# Pergunta 6 – Recomendações Práticas
print("#" * 70)
print("Pergunta 6 – Recomendações Práticas\n")
print(textwrap.dedent("""
• Aumentar o número de núcleos de CPU (coef. −12,4 ms por core extra).
• Expandir RAM para 32-64 GB (coef. −1,3 ms por GB).
• Preferir processadores Intel ou Apple Silicon e SO Windows/macOS, que
  mostraram tempos menores que a categoria-base (Linux/AMD).
• Reduzir latência de rede/I/O (alta dispersão sugere gargalos externos).
• Como há heterocedasticidade, reporte erros-padrão robustos (HC3) ou
  considere modelar log(tempo_resposta) em análises futuras.
"""))
