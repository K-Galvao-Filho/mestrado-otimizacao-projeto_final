import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD
import streamlit as st
import os

# -----------------------------
# PARÂMETROS GERAIS
# -----------------------------

NUMERO_CONSUMIDORES = 10
PERIODOS_DIA = 24
SEMENTE_ALEATORIA = 42
np.random.seed(SEMENTE_ALEATORIA)

# -----------------------------
# DADOS REALISTAS (ALAGOAS)
# -----------------------------

def carregar_dados_alagoas():
    GERACAO_SOLAR = np.array([
        0, 0, 0, 0, 0, 0.5, 2, 10, 30, 50, 70, 80, 75, 60, 40, 20, 5, 1, 0, 0, 0, 0, 0, 0
    ]) * 0.5
    DEMANDA_HORARIA = np.zeros((NUMERO_CONSUMIDORES, PERIODOS_DIA))
    for i in range(NUMERO_CONSUMIDORES):
        for t in range(PERIODOS_DIA):
            if 18 <= t <= 22:
                DEMANDA_HORARIA[i, t] = np.random.uniform(0.8, 1.5)
            elif 8 <= t <= 17:
                DEMANDA_HORARIA[i, t] = np.random.uniform(0.3, 0.8)
            else:
                DEMANDA_HORARIA[i, t] = np.random.uniform(0.2, 0.4)
    PRIORIDADES = np.random.uniform(0.05, 0.2, NUMERO_CONSUMIDORES)
    PRIORIDADES = PRIORIDADES / PRIORIDADES.sum()
    PERDAS_REDE = 0.074
    LIMITE_INJECAO = 0.8 * GERACAO_SOLAR
    CUSTO_DISPONIBILIDADE = 30
    CUSTO_OPERACIONAL = 0.1
    CUSTO_REDE = 0.8
    FATOR_EMISSAO = 0.4
    DEMANDA_MINIMA_PRIORITARIOS = 5
    return (GERACAO_SOLAR, DEMANDA_HORARIA, PRIORIDADES, PERDAS_REDE,
            LIMITE_INJECAO, CUSTO_DISPONIBILIDADE, CUSTO_OPERACIONAL,
            CUSTO_REDE, FATOR_EMISSAO, DEMANDA_MINIMA_PRIORITARIOS)

(GERACAO_SOLAR, DEMANDA_HORARIA, PRIORIDADES, PERDAS_REDE, LIMITE_INJECAO,
 CUSTO_DISPONIBILIDADE, CUSTO_OPERACIONAL, CUSTO_REDE, FATOR_EMISSAO,
 DEMANDA_MINIMA_PRIORITARIOS) = carregar_dados_alagoas()

CONSUMIDORES_PRIORITARIOS = sorted(range(NUMERO_CONSUMIDORES),
                                   key=lambda i: -PRIORIDADES[i])[:NUMERO_CONSUMIDORES//2]

# -----------------------------
# CENÁRIO SEQUENCIAL
# -----------------------------

def cenario_sequencial():
    alocacao = np.zeros((NUMERO_CONSUMIDORES, PERIODOS_DIA))
    creditos = np.zeros((NUMERO_CONSUMIDORES, PERIODOS_DIA))
    energia_rede = np.zeros((NUMERO_CONSUMIDORES, PERIODOS_DIA))
    for t in range(PERIODOS_DIA):
        energia_disponivel = GERACAO_SOLAR[t] * (1 - PERDAS_REDE)
        for i in range(NUMERO_CONSUMIDORES):
            if energia_disponivel >= DEMANDA_HORARIA[i, t]:
                alocacao[i, t] = 1
                energia_disponivel -= DEMANDA_HORARIA[i, t]
                creditos[i, t] = max(0, GERACAO_SOLAR[t] * alocacao[i, t] -
                                     DEMANDA_HORARIA[i, t])
            else:
                energia_rede[i, t] = DEMANDA_HORARIA[i, t] * (1 - alocacao[i, t])
    return alocacao, creditos, energia_rede

# -----------------------------
# MODELO DE OTIMIZAÇÃO
# -----------------------------

def criar_modelo_otimizacao(modo):
    modelo = LpProblem("Alocacao_Solar_Alagoas", LpMaximize)
    alocacao = LpVariable.dicts(
        "alocacao",
        ((i, t) for i in range(NUMERO_CONSUMIDORES) for t in range(PERIODOS_DIA)),
        cat=LpBinary
    )
    creditos = LpVariable.dicts(
        "creditos",
        ((i, t) for i in range(NUMERO_CONSUMIDORES) for t in range(PERIODOS_DIA)),
        lowBound=0
    )
    energia_rede = LpVariable.dicts(
        "energia_rede",
        ((i, t) for i in range(NUMERO_CONSUMIDORES) for t in range(PERIODOS_DIA)),
        lowBound=0
    )
    if modo == "prioritario":
        modelo += (
            0.6 * lpSum(PRIORIDADES[i] * alocacao[i, t] * DEMANDA_HORARIA[i, t]
                        for i in range(NUMERO_CONSUMIDORES) for t in range(PERIODOS_DIA)) +
            0.3 * lpSum(creditos[i, t]
                        for i in range(NUMERO_CONSUMIDORES) for t in range(PERIODOS_DIA)) -
            0.1 * lpSum(CUSTO_REDE * energia_rede[i, t]
                        for i in range(NUMERO_CONSUMIDORES) for t in range(PERIODOS_DIA))
        )
    else:
        modelo += (
            lpSum(alocacao[i, t] * DEMANDA_HORARIA[i, t]
                  for i in range(NUMERO_CONSUMIDORES) for t in range(PERIODOS_DIA)) -
            0.1 * lpSum(energia_rede[i, t]
                        for i in range(NUMERO_CONSUMIDORES) for t in range(PERIODOS_DIA))
        )
    for t in range(PERIODOS_DIA):
        modelo += (
            lpSum(DEMANDA_HORARIA[i, t] * alocacao[i, t]
                  for i in range(NUMERO_CONSUMIDORES)) <=
            GERACAO_SOLAR[t] * (1 - PERDAS_REDE)
        )
    for i in range(NUMERO_CONSUMIDORES):
        for t in range(PERIODOS_DIA):
            modelo += (
                creditos[i, t] <=
                GERACAO_SOLAR[t] * alocacao[i, t] - DEMANDA_HORARIA[i, t]
            )
            modelo += creditos[i, t] >= 0
    for t in range(PERIODOS_DIA):
        modelo += (
            lpSum(creditos[i, t] for i in range(NUMERO_CONSUMIDORES)) <=
            LIMITE_INJECAO[t]
        )
    for i in range(NUMERO_CONSUMIDORES):
        for t in range(PERIODOS_DIA):
            modelo += (
                DEMANDA_HORARIA[i, t] * alocacao[i, t] + energia_rede[i, t] ==
                DEMANDA_HORARIA[i, t]
            )
    if modo == "prioritario":
        for i in CONSUMIDORES_PRIORITARIOS:
            modelo += (
                lpSum(DEMANDA_HORARIA[i, t] * alocacao[i, t]
                      for t in range(PERIODOS_DIA)) >=
                DEMANDA_MINIMA_PRIORITARIOS
            )
    return modelo, alocacao, creditos, energia_rede

# -----------------------------
# EXECUÇÃO DOS CENÁRIOS
# -----------------------------

def executar_cenarios():
    resultados = {}
    aloc_seq, cred_seq, rede_seq = cenario_sequencial()
    resultados["Sequencial"] = {
        "alocacao": aloc_seq * DEMANDA_HORARIA,
        "creditos": cred_seq,
        "energia_rede": rede_seq
    }
    modelo_simples, aloc_simples, cred_simples, rede_simples = criar_modelo_otimizacao("simples")
    solver = PULP_CBC_CMD(msg=0)
    modelo_simples.solve(solver)
    aloc_2 = np.array([[aloc_simples[i, t].varValue for t in range(PERIODOS_DIA)]
                       for i in range(NUMERO_CONSUMIDORES)])
    cred_2 = np.array([[cred_simples[i, t].varValue for t in range(PERIODOS_DIA)]
                       for i in range(NUMERO_CONSUMIDORES)])
    rede_2 = np.array([[rede_simples[i, t].varValue for t in range(PERIODOS_DIA)]
                       for i in range(NUMERO_CONSUMIDORES)])
    resultados["PLI_Simples"] = {
        "alocacao": aloc_2 * DEMANDA_HORARIA,
        "creditos": cred_2,
        "energia_rede": rede_2
    }
    modelo_prior, aloc_prior, cred_prior, rede_prior = criar_modelo_otimizacao("prioritario")
    modelo_prior.solve(solver)
    aloc_3 = np.array([[aloc_prior[i, t].varValue for t in range(PERIODOS_DIA)]
                       for i in range(NUMERO_CONSUMIDORES)])
    cred_3 = np.array([[cred_prior[i, t].varValue for t in range(PERIODOS_DIA)]
                       for i in range(NUMERO_CONSUMIDORES)])
    rede_3 = np.array([[rede_prior[i, t].varValue for t in range(PERIODOS_DIA)]
                       for i in range(NUMERO_CONSUMIDORES)])
    resultados["PLI_Prioritario"] = {
        "alocacao": aloc_3 * DEMANDA_HORARIA,
        "creditos": cred_3,
        "energia_rede": rede_3
    }
    return resultados

# -----------------------------
# ANÁLISE DE RESULTADOS
# -----------------------------

def analisar_resultados(cenario_nome, alocacao, creditos, energia_rede):
    energia_alocada = alocacao.sum()
    eficiencia = (energia_alocada / (GERACAO_SOLAR.sum() * (1 - PERDAS_REDE)) * 100
                  if GERACAO_SOLAR.sum() > 0 else 0)
    autossuficiencia = (energia_alocada / DEMANDA_HORARIA.sum() * 100
                        if DEMANDA_HORARIA.sum() > 0 else 0)
    equidade = alocacao.sum(axis=1).std()
    emissoes_evitadas = energia_alocada * FATOR_EMISSAO
    custo_total = (energia_rede.sum() * CUSTO_REDE +
                   CUSTO_DISPONIBILIDADE * NUMERO_CONSUMIDORES / 30 +
                   alocacao.sum() * CUSTO_OPERACIONAL)
    creditos_total = creditos.sum()
    print(f"\n--- {cenario_nome} ---")
    print(f"Energia Alocada: {energia_alocada:.2f} kWh")
    print(f"Eficiência: {eficiencia:.2f}%")
    print(f"Autossuficiência: {autossuficiencia:.2f}%")
    print(f"Equidade (Desvio Padrão): { hypothalamus:.2f}")
    print(f"Emissões Evitadas: {emissoes_evitadas:.2f} kg CO2")
    print(f"Custo Total: R$ {custo_total:.2f}")
    print(f"Créditos Gerados: {creditos_total:.2f} kWh")
    return {
        "cenario": cenario_nome,
        "energia_alocada": energia_alocada,
        "eficiencia": eficiencia,
        "autossuficiencia": autossuficiencia,
        "equidade": equidade,
        "emissoes_evitadas": emissoes_evitadas,
        "custo_total": custo_total,
        "creditos": creditos_total,
        "energia_por_unidade": alocacao.sum(axis=1)
    }

# -----------------------------
# VISUALIZAÇÕES
# -----------------------------

def gerar_visualizacoes(resultados):
    plt.figure(figsize=(12, 6))
    for nome, dados in resultados.items():
        plt.plot(dados["energia_por_unidade"], label=nome)
    plt.xlabel("Consumidor")
    plt.ylabel("Energia Alocada (kWh)")
    plt.title("Distribuição de Energia Solar por Consumidor")
    plt.legend()
    plt.grid(True)
    plt.savefig("saidas/energia_por_consumidor.png")
    plt.close()
    df_indicadores = pd.DataFrame([
        {
            "Cenário": dados["cenario"],
            "Eficiência (%)": dados["eficiencia"],
            "Autossuficiência (%)": dados["autossuficiencia"],
            "Equidade": dados["equidade"],
            "Emissões Evitadas (kg CO2)": dados["emissoes_evitadas"],
            "Custo Total (R$)": dados["custo_total"],
            "Créditos (kWh)": dados["creditos"]
        }
        for dados in resultados.values()
    ])
    df_indicadores.plot(kind="bar", x="Cenário", figsize=(12, 8))
    plt.title("Comparação de Indicadores")
    plt.grid(axis="y")
    plt.savefig("saidas/indicadores.png")
    plt.close()
    df_indicadores.to_csv("saidas/indicadores.csv", index=False)
    for nome, dados in resultados.items():
        df_aloc = pd.DataFrame(dados["alocacao"],
                              columns=[f"Hora_{t}" for t in range(PERIODOS_DIA)])
        df_aloc.index.name = "Consumidor"
        df_aloc.to_csv(f"saidas/alocacao_{nome.lower().replace(' ', '_')}.csv")
    return df_indicadores

# -----------------------------
# DASHBOARD INTERATIVO
# -----------------------------

def exibir_dashboard(resultados):
    st.title("Comunidade Solar em Alagoas")
    st.write("Otimização de Energia Solar com Base na Legislação da ANEEL")
    cenario = st.selectbox("Selecione o Cenário", list(resultados.keys()))
    dados = resultados[cenario]
    st.write(f"**Energia Alocada**: {dados['energia_alocada']:.2f} kWh")
    st.write(f"**Eficiência**: {dados['eficiencia']:.2f}%")
    st.write(f"**Autossuficiência**: {dados['autossuficiencia']:.2f}%")
    st.write(f"**Equidade (Desvio Padrão)**: {dados['equidade']:.2f}")
    st.write(f"**Emissões Evitadas**: {dados['emissoes_evitadas']:.2f} kg CO2")
    st.write(f"**Custo Total**: R$ {dados['custo_total']:.2f}")
    st.write(f"**Créditos Gerados**: {dados['creditos']:.2f} kWh")
    st.image("saidas/energia_por_consumidor.png")
    st.image("saidas/indicadores.png")
    st.write(f"Alocação Detalhada - {cenario}")
    df_aloc = pd.DataFrame(dados["alocacao"],
                          columns=[f"Hora_{t}" for t in range(PERIODOS_DIA)])
    df_aloc.index.name = "Consumidor"
    st.dataframe(df_aloc)

# -----------------------------
# EXECUÇÃO PRINCIPAL
# -----------------------------

if __name__ == "__main__":
    os.makedirs("saidas", exist_ok=True)
    resultados_cenarios = executar_cenarios()
    analises = {}
    for nome, dados in resultados_cenarios.items():
        analises[nome] = analisar_resultados(nome, dados["alocacao"],
                                            dados["creditos"],
                                            dados["energia_rede"])
    gerar_visualizacoes(analises)
    exibir_dashboard(analises)