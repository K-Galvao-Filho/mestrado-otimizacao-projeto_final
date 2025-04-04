import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD

# -----------------------------
# PARÂMETROS GERAIS
# -----------------------------

# Define o número de unidades consumidoras (ex: casas, residências ou pontos de consumo)
N = 10  # Total de consumidores da comunidade

# Define o número de períodos de tempo (um dia dividido em 24 horas)
T = 24  # Total de períodos (horas do dia)

# Define a semente do gerador de números aleatórios para garantir reprodutibilidade
# Isso assegura que, ao executar o código novamente, os mesmos valores aleatórios serão gerados
np.random.seed(42)

# ---------------------------------------------------------
# GERAÇÃO DE DADOS ALEATÓRIOS PARA A SIMULAÇÃO
# ---------------------------------------------------------

# Et: vetor com a energia solar disponível em cada período do dia (em unidades arbitrárias)
# Durante os horários de pico solar (entre 10h e 16h), a energia é maior (15 a 30)
# Fora desses horários, a geração é mais baixa (5 a 15)
Et = [
    np.random.randint(15, 30) if 10 <= t <= 16 else np.random.randint(5, 15)
    for t in range(T)
]

# di_t: matriz (N x T) com a demanda de energia de cada unidade em cada período
# Valores variam entre 1 e 4 unidades de energia por hora, gerando perfis variados
di_t = np.random.randint(1, 5, size=(N, T))

# pi: vetor com o nível de prioridade de cada unidade consumidora
# Os valores vão de 1 (baixa prioridade) a 5 (alta prioridade), simulando desigualdade social
pi = np.random.randint(1, 6, size=N)

# prioridade_ids: índice das unidades com maior prioridade (metade dos consumidores)
# São escolhidos os 50% com maiores valores de prioridade
prioridade_ids = sorted(range(N), key=lambda i: -pi[i])[:N//2]

# Dmin: valor mínimo de energia (acumulada ao longo do dia) que as unidades prioritárias devem receber
# Esse parâmetro é usado como restrição no modelo do Cenário 3
Dmin = 30


# -----------------------------
# CENÁRIO 1: Distribuição Igualitária
# -----------------------------

# Inicializa uma matriz 10x24 (N x T) com zeros, representando a alocação de energia
# Cada posição igualitaria[i][t] indica se a unidade 'i' foi atendida no período 't'
igualitaria = np.zeros((N, T))

# Alocação sequencial da energia disponível em cada período (cenário 1: algoritmo guloso)
for t in range(T): 
    energia_restante = Et[t]  # Energia solar disponível no período 't'
    i = 0  # Inicia pela primeira unidade consumidora
    while energia_restante > 0 and i < N:
        demanda = di_t[i][t]  # Demanda da unidade 'i' no período 't'
        if energia_restante >= demanda:
            igualitaria[i][t] = 1  # Marca que a unidade 'i' será atendida nesse período
            energia_restante -= demanda  # Subtrai a demanda da energia disponível
        i += 1  # Passa para a próxima unidade consumidora

# Converte a matriz 'igualitaria' para inteiros (0 ou 1), indicando decisão binária
res1 = np.array([[int(igualitaria[i][t]) for t in range(T)] for i in range(N)])

# Calcula a matriz final de energia alocada: multiplica a decisão binária pela demanda
# Se igualitaria[i][t] = 1, então a energia alocada é igual à demanda di_t[i][t]
# Se for 0, não houve atendimento à unidade naquele horário
aloc_1 = igualitaria * di_t

# -----------------------------
# CENÁRIO 2: PLI sem prioridade
# -----------------------------

# Define o problema de otimização como um modelo de Programação Linear Inteira (PLI)
# Nome do modelo: "Cenario2_PLI_Sem_Prioridade", tipo: maximização
model2 = LpProblem("Cenario2_PLI_Sem_Prioridade", LpMaximize)

# Cria as variáveis de decisão binárias x2[i,t]
# x2[i,t] = 1 se a unidade i for atendida no período t; 0 caso contrário
x2 = LpVariable.dicts("x2", ((i, t) for i in range(N) for t in range(T)), cat=LpBinary)

# Define a função objetivo: maximizar o número total de atendimentos
# Ou seja, somar todos os x2[i,t] (quantas vezes uma unidade é atendida)
model2 += lpSum(x2[i, t] for i in range(N) for t in range(T))

# Adiciona restrições de capacidade para cada período t
# A soma das demandas das unidades atendidas em t não pode ultrapassar a energia disponível Et[t]
for t in range(T):
    model2 += lpSum(di_t[i][t] * x2[i, t] for i in range(N)) <= Et[t]

# Resolve o modelo utilizando o solucionador CBC (Coin-or Branch and Cut)
# A opção msg=0 suprime as mensagens durante a resolução
model2.solve(PULP_CBC_CMD(msg=0))

# Extrai os valores otimizados das variáveis de decisão (0 ou 1)
# res2[i][t] = 1 se a unidade i foi atendida no período t, conforme resultado da otimização
res2 = np.array([[int(x2[i, t].varValue) for t in range(T)] for i in range(N)])

# Calcula a matriz final de energia alocada multiplicando a decisão binária pela demanda
# Isso gera a quantidade de energia efetivamente entregue a cada unidade em cada período
aloc_2 = res2 * di_t

# -----------------------------
# CENÁRIO 3: PLI com prioridade
# -----------------------------

# Define o modelo de Programação Linear Inteira (PLI) com prioridade socioeconômica
# Nome do modelo: "Cenario3_PLI_Com_Prioridade", tipo: maximização
model3 = LpProblem("Cenario3_PLI_Com_Prioridade", LpMaximize)

# Cria variáveis de decisão binárias x3[i,t]
# x3[i,t] = 1 indica que a unidade i será atendida no período t
x3 = LpVariable.dicts("x3", ((i, t) for i in range(N) for t in range(T)), cat=LpBinary)

# Define a função objetivo ponderada: maximizar o atendimento total,
# atribuindo maior peso às unidades com maior prioridade (pi[i])
model3 += lpSum(pi[i] * x3[i, t] for i in range(N) for t in range(T))

# Adiciona as restrições técnicas de capacidade energética por período
# A energia alocada às unidades em cada período t não pode ultrapassar a geração disponível Et[t]
for t in range(T):
    model3 += lpSum(di_t[i][t] * x3[i, t] for i in range(N)) <= Et[t]

# Adiciona restrições de justiça: cada unidade considerada prioritária deve receber,
# no total, ao menos uma quantidade mínima de energia (Dmin) ao longo dos 24 períodos
for i in prioridade_ids:
    model3 += lpSum(di_t[i][t] * x3[i, t] for t in range(T)) >= Dmin

# Resolve o modelo utilizando o solver CBC, de forma silenciosa (msg=0)
model3.solve(PULP_CBC_CMD(msg=0))

# Extrai os resultados otimizados: 0 ou 1 para cada variável x3[i,t]
res3 = np.array([[int(x3[i, t].varValue) for t in range(T)] for i in range(N)])

# Calcula a matriz final de energia alocada: multiplica decisão binária pela demanda
# aloc_3[i][t] = di_t[i][t] se x3[i,t] = 1; caso contrário, 0
aloc_3 = res3 * di_t

# -----------------------------
# FUNÇÃO DE ANÁLISE
# -----------------------------

def analisar(aloc, nome):
    # Função que calcula indicadores de desempenho da alocação de energia
    # aloc: matriz (N x T) com a energia efetivamente alocada a cada unidade em cada período
    # nome: nome do cenário a ser analisado (string)

    # Soma de toda a energia alocada (valor total atendido)
    energia_total = aloc.sum()
    print(f"\nEnergia total alocada - {nome}: {energia_total}")

    # Eficiência: razão entre energia alocada e energia solar disponível total
    # Mede o aproveitamento da energia gerada
    eficiencia = energia_total / sum(Et) * 100
    print(f"Eficiência - {nome}: {eficiencia:.2f}%")

    # Autossuficiência: razão entre energia alocada e demanda total
    # Mede o quanto da demanda da comunidade foi efetivamente atendido
    autossuf = energia_total / di_t.sum() * 100
    print(f"Autossuficiência - {nome}: {autossuf:.2f}%")

    # Equidade: medida estatística baseada no desvio padrão da energia total recebida por unidade
    # Quanto menor o desvio, mais homogênea e justa é a distribuição
    eq_std = aloc.sum(axis=1).std()
    print(f"Equidade (Desvio Padrão) - {nome}: {eq_std:.2f}")

    # Retorna um dicionário com os resultados calculados, útil para gerar gráficos e tabelas
    return {
        "cenário": nome,
        "eficiência (%)": eficiencia,
        "autossuficiência (%)": autossuf,
        "desvio padrão": eq_std,
        "energia_unidade": aloc.sum(axis=1)  # vetor com energia total recebida por cada unidade
    }

# -----------------------------
# ANÁLISE DOS RESULTADOS
# -----------------------------

# Analisando os resultados de cada cenário com a função 'analisar'
# Cada item da lista será um dicionário com os indicadores de desempenho

# Cenário 1: Alocação sequencial simples (algoritmo guloso)
# Cenário 2: Otimização por PLI sem pesos sociais
# Cenário 3: Otimização por PLI com prioridade e meta mínima para unidades vulneráveis
analises = [
    analisar(aloc_1, "Cenário 1"),
    analisar(aloc_2, "Cenário 2"),
    analisar(aloc_3, "Cenário 3"),
]

# -----------------------------
# CRIAÇÃO DO DATAFRAME PARA AS MÉTRICAS
# -----------------------------

# Cria um DataFrame com os indicadores principais de cada cenário
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

df_res1 = pd.DataFrame(res1, columns=[f"T{t}" for t in range(T)])
df_res1.index.name = "Unidade"
print("Cenário 1 - PLI sem prioridade:")
print(df_res1)

df_aloc2 = pd.DataFrame(aloc_2, columns=[f"T{t}" for t in range(T)])
df_aloc2.index.name = "Unidade"
print("\nEnergia alocada - Cenário 2:")
print(df_aloc2)

df_res2 = pd.DataFrame(res2, columns=[f"T{t}" for t in range(T)])
df_res2.index.name = "Unidade"
print("Cenário 2 - PLI sem prioridade:")
print(df_res2)

df_res3 = pd.DataFrame(res3, columns=[f"T{t}" for t in range(T)])
df_res3.index.name = "Unidade"
print("\nCenário 3 - PLI com prioridade:")
print(df_res3)

df_aloc3 = pd.DataFrame(aloc_3, columns=[f"T{t}" for t in range(T)])
df_aloc3.index.name = "Unidade"
print("\nEnergia alocada - Cenário 3:")
print(df_aloc3)

#métricas
df_metricas.to_csv("saidas/metricas.csv", index=False)

df_aloc1.to_csv("saidas/demanda_cenario1.csv")
df_res1.to_csv("saidas/alocacao_cenario1.csv")

df_aloc2.to_csv("saidas/demanda_cenario2.csv")
df_res2.to_csv("saidas/alocacao_cenario2.csv")

df_aloc3.to_csv("saidas/demanda_cenario3.csv")
df_res3.to_csv("saidas/alocacao_cenario3.csv")