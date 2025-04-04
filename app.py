import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD

# -----------------------------
# PARÂMETROS GERAIS
# -----------------------------
N = 10 # Número de unidades consumidoras
T = 24 # Número de períodos de tempo (horas)

# Definindo a semente para reprodutibilidade
np.random.seed(42)

# GERAÇÃO DE DADOS ALEATÓRIOS
# Et: energia disponível em cada período
Et = [np.random.randint(15, 30) if 10 <= t <= 16 else np.random.randint(5, 15) for t in range(T)]
# di_t: demanda de energia de cada unidade em cada período
di_t = np.random.randint(1, 5, size=(N, T))
# pi: prioridade de cada unidade
pi = np.random.randint(1, 6, size=N)
# prioridade_ids: IDs das unidades com prioridade
prioridade_ids = sorted(range(N), key=lambda i: -pi[i])[:N//2]
# Dmin: demanda mínima para unidades prioritárias
Dmin = 30

# -----------------------------
# CENÁRIO 1: Distribuição Igualitária
# -----------------------------

igualitaria = np.zeros((N, T)) # Inicializa a matriz de alocação
for t in range(T): 
    energia_restante = Et[t]
    i = 0
    while energia_restante > 0 and i < N:
        demanda = di_t[i][t]
        if energia_restante >= demanda:
            igualitaria[i][t] = 1
            energia_restante -= demanda
        i += 1
res1 = np.array([[int(igualitaria[i][t]) for t in range(T)] for i in range(N)])
aloc_1 = igualitaria * di_t # Alocação de energia

# -----------------------------
# CENÁRIO 2: PLI sem prioridade
# -----------------------------
model2 = LpProblem("Cenario2_PLI_Sem_Prioridade", LpMaximize)
x2 = LpVariable.dicts("x2", ((i, t) for i in range(N) for t in range(T)), cat=LpBinary)
model2 += lpSum(x2[i, t] for i in range(N) for t in range(T))

for t in range(T):
    model2 += lpSum(di_t[i][t] * x2[i, t] for i in range(N)) <= Et[t]

model2.solve(PULP_CBC_CMD(msg=0))

res2 = np.array([[int(x2[i, t].varValue) for t in range(T)] for i in range(N)])
aloc_2 = res2 * di_t

# -----------------------------
# CENÁRIO 3: PLI com prioridade
# -----------------------------
model3 = LpProblem("Cenario3_PLI_Com_Prioridade", LpMaximize)
x3 = LpVariable.dicts("x3", ((i, t) for i in range(N) for t in range(T)), cat=LpBinary)
model3 += lpSum(pi[i] * x3[i, t] for i in range(N) for t in range(T))

for t in range(T):
    model3 += lpSum(di_t[i][t] * x3[i, t] for i in range(N)) <= Et[t]
for i in prioridade_ids:
    model3 += lpSum(di_t[i][t] * x3[i, t] for t in range(T)) >= Dmin

model3.solve(PULP_CBC_CMD(msg=0))

res3 = np.array([[int(x3[i, t].varValue) for t in range(T)] for i in range(N)])
aloc_3 = res3 * di_t

# -----------------------------
# FUNÇÃO DE ANÁLISE
# -----------------------------
def analisar(aloc, nome):
    # Energia total alocada por unidade
    energia_total = aloc.sum()
    print(f"\nEnergia total alocada - {nome}: {energia_total}")
    
    # Eficiência: energia total alocada / energia total disponível
    eficiencia = energia_total / sum(Et) * 100
    print(f"Eficiência - {nome}: {eficiencia:.2f}%")
    
    # Autossuficiência: energia total alocada / demanda total 
    autossuf = energia_total / di_t.sum() * 100
    print(f"Autossuficiência - {nome}: {autossuf:.2f}%")
    
    # Desvio padrão: medida de equidade
    # quanto menor o desvio padrão, mais equitativa é a alocação
    eq_std = aloc.sum(axis=1).std()
    return {
        "cenário": nome,
        "eficiência (%)": eficiencia,
        "autossuficiência (%)": autossuf,
        "desvio padrão": eq_std,
        "energia_unidade": aloc.sum(axis=1)
    }

analises = [
    analisar(aloc_1, "Cenário 1"),
    analisar(aloc_2, "Cenário 2"),
    analisar(aloc_3, "Cenário 3"),
]

df_metricas = pd.DataFrame([{
    "Cenário": a["cenário"],
    "Eficiência (%)": a["eficiência (%)"],
    "Autossuficiência (%)": a["autossuficiência (%)"],
    "Equidade (Desvio Padrão)": a["desvio padrão"]
} for a in analises])

# -----------------------------
# VISUALIZAÇÃO DOS RESULTADOS
# -----------------------------
plt.figure(figsize=(12, 6))
for a in analises:
    plt.plot(a["energia_unidade"], label=a["cenário"])
plt.xlabel("Unidade Consumidora")
plt.ylabel("Energia Total Recebida")
plt.title("Distribuição de Energia por Unidade")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df_metricas.plot(kind="bar", x="Cenário", figsize=(10, 6))
plt.title("Comparação de Eficiência, Autossuficiência e Equidade")
plt.ylabel("Valor (%) / Desvio Padrão")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# -----------------------------
# SALVANDO RESULTADOS EM CSV
# -----------------------------

df_aloc1 = pd.DataFrame(aloc_1, columns=[f"T{t}" for t in range(T)])
df_aloc1.index.name = "Unidade"
print("\nEnergia alocada - Cenário 1:")
print(df_aloc1)
df_aloc1.to_csv("demanda_cenario1.csv")

df_res1 = pd.DataFrame(res1, columns=[f"T{t}" for t in range(T)])
df_res1.index.name = "Unidade"
print("Cenário 1 - PLI sem prioridade:")
print(df_res1)
df_res1.to_csv("alocacao_cenario1.csv")

df_aloc2 = pd.DataFrame(aloc_2, columns=[f"T{t}" for t in range(T)])
df_aloc2.index.name = "Unidade"
print("\nEnergia alocada - Cenário 2:")
print(df_aloc2)
df_aloc2.to_csv("demanda_cenario2.csv")

df_res2 = pd.DataFrame(res2, columns=[f"T{t}" for t in range(T)])
df_res2.index.name = "Unidade"
print("Cenário 2 - PLI sem prioridade:")
print(df_res2)
df_aloc2.to_csv("alocacao_cenario2.csv")

df_res3 = pd.DataFrame(res3, columns=[f"T{t}" for t in range(T)])
df_res3.index.name = "Unidade"
print("\nCenário 3 - PLI com prioridade:")
print(df_res3)
df_res3.to_csv("demanda_cenario3.csv")

df_aloc3 = pd.DataFrame(aloc_3, columns=[f"T{t}" for t in range(T)])
df_aloc3.index.name = "Unidade"
print("\nEnergia alocada - Cenário 3:")
print(df_aloc3)
df_aloc3.to_csv("alocacao_cenario3.csv")