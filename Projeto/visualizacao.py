import numpy as np
import matplotlib.pyplot as plt

def plotar_graficos(analises, aloc_list, rede_list, params):
    """Gera gráficos para o Cenário 3."""
    num_residencias = params['num_residencias']
    num_horas = params['num_horas']
    tarifas_residencias = params['tarifas_residencias']
    
    # Extrair dados do Cenário 3 (único cenário)
    analise = analises[0]
    aloc = aloc_list[0]
    rede = rede_list[0]

    # Gráfico 1: Métricas do Cenário 3
    metricas = ["Eficiência (%)", "Autossuficiência (%)", "Desvio Padrão (kWh)", "Custo (R$)"]
    valores = [
        analise["eficiência (%)"],
        analise["autossuficiência (%)"],
        analise["desvio padrão"],
        analise["custo (R$)"]
    ]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metricas, valores)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}", ha='center', va='bottom')
    plt.title("Métricas do Cenário 3")
    plt.ylabel("Valor")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("saidas/metricas_cenario3.png")
    plt.close()

    # Gráfico 2: Boxplot de energia por unidade
    energia_unidade = analise["energia_unidade"]
    plt.figure(figsize=(8, 5))
    plt.boxplot(energia_unidade, labels=["Cenário 3"])
    plt.text(1, np.median(energia_unidade), f"{np.median(energia_unidade):.2f}", ha='center', va='bottom', color='red')
    plt.ylabel("Energia Total Recebida (kWh)")
    plt.title("Distribuição de Energia por Unidade (Cenário 3)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("saidas/boxplot_cenario3.png")
    plt.close()

    # Gráfico 3: Custo acumulado por hora
    custo_hora = np.cumsum([sum(tarifas_residencias[i][t] * rede[i][t] for i in range(num_residencias)) for t in range(num_horas)])
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_horas), custo_hora, label="Cenário 3", marker="^")
    for t in range(0, num_horas, 6):
        plt.text(t, custo_hora[t], f"{custo_hora[t]:.2f}", ha='center', va='bottom')
    plt.xlabel("Hora")
    plt.ylabel("Custo Acumulado (R$)")
    plt.title("Custo Acumulado da Rede (Cenário 3)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("saidas/custo_horario_cenario3.png")
    plt.close()

    # Gráfico 4: Energia Tarifa Social vs. Demais
    ts_indices = [i for i in range(num_residencias) if i % 2 == 0]
    nao_ts_indices = [i for i in range(num_residencias) if i % 2 != 0]
    energia_ts = sum(energia_unidade[i] for i in ts_indices)
    energia_nao_ts = sum(energia_unidade[i] for i in nao_ts_indices)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(["Tarifa Social", "Demais"], [energia_ts, energia_nao_ts])
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}", ha='center', va='bottom')
    plt.ylabel("Energia Alocada (kWh)")
    plt.title("Energia Alocada: Tarifa Social vs. Demais (Cenário 3)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("saidas/energia_ts_vs_demais_cenario3.png")
    plt.close()