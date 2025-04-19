import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import configurar_parametros
from cenario1 import cenario_1_proporcional_simples
from cenario2 import cenario_2_priorizacao_tarifa_social
from cenario3 import cenario_3_multiobjetivo
from analise import analisar_resultados

def exportar_resultados(analises, aloc_list, params):
    """Exporta resultados para CSV."""
    os.makedirs("saidas", exist_ok=True)

    df_comparacao = pd.DataFrame(analises)
    df_comparacao.to_csv("saidas/comparacao_todos_cenarios.csv", index=False)

    for i, aloc in enumerate(aloc_list):
        df_aloc = pd.DataFrame(
            aloc,
            columns=[f"Hora {t}" for t in range(params['num_horas'])],
            index=[f"Unidade {j+1}{' (TS)' if j % 2 == 0 else ''}" for j in range(params['num_residencias'])]
        )
        df_aloc.to_csv(f"saidas/alocacao_cenario{i+1}.csv")

def plotar_graficos(analises, aloc_list, rede_list, params):
    """Gera gráficos comparativos para os três cenários."""
    os.makedirs("saidas", exist_ok=True)
    num_cenarios = len(analises)

    # Gráfico 1: Comparativo de Eficiência, Autossuficiência, Desvio Padrão, Custo, Uso da Rede
    metricas = ["eficiência (%)", "autossuficiência (%)", "desvio padrão", "custo (R$)", "rede (kWh)"]
    for metrica in metricas:
        plt.figure(figsize=(8,5))
        valores = [analise[metrica] for analise in analises]
        labels = [analise["cenário"] for analise in analises]
        bars = plt.bar(labels, valores)
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}", ha='center', va='bottom')
        plt.title(f"Comparação da métrica: {metrica}")
        plt.ylabel(metrica)
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(f"saidas/comparacao_{metrica.replace(' ', '_').replace('(%)','')}.png")
        plt.close()

    # Gráfico 2: Boxplot de energia recebida por unidade
    plt.figure(figsize=(10,6))
    data = [analise["energia_unidade"] for analise in analises]
    labels = [analise["cenário"] for analise in analises]
    plt.boxplot(data, labels=labels)
    plt.title("Distribuição de Energia Recebida por Unidade")
    plt.ylabel("Energia Total Recebida (kWh)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("saidas/boxplot_energia_unidades.png")
    plt.close()

    # Gráfico 3: Energia Tarifa Social vs Demais por Cenário
    num_residencias = params['num_residencias']
    ts_indices = [i for i in range(num_residencias) if i % 2 == 0]
    nao_ts_indices = [i for i in range(num_residencias) if i % 2 != 0]

    energia_ts = []
    energia_nao_ts = []

    for analise in analises:
        energia_unidade = analise["energia_unidade"]
        energia_ts.append(sum(energia_unidade[i] for i in ts_indices))
        energia_nao_ts.append(sum(energia_unidade[i] for i in nao_ts_indices))

    x = np.arange(len(analises))
    width = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, energia_ts, width, label='Tarifa Social')
    plt.bar(x + width/2, energia_nao_ts, width, label='Demais')

    plt.xticks(x, [analise["cenário"] for analise in analises])
    plt.ylabel("Energia Alocada (kWh)")
    plt.title("Energia Alocada: Tarifa Social vs Demais por Cenário")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("saidas/energia_ts_vs_demais_cenarios.png")
    plt.close()

def main():
    os.makedirs("saidas", exist_ok=True)
    params = configurar_parametros()

    analises = []
    aloc_list = []
    rede_list = []

    # Executar Cenário 1
    print("\nExecutando Cenário 1: Proporcional Simples...")
    aloc1, rede1, excedente1 = cenario_1_proporcional_simples(params)
    analise1 = analisar_resultados(aloc1, rede1, params, "Cenário 1 - Proporcional Simples")
    analises.append(analise1)
    aloc_list.append(aloc1)
    rede_list.append(rede1)

    # Executar Cenário 2
    print("\nExecutando Cenário 2: Priorização Tarifa Social...")
    aloc2, rede2, excedente2 = cenario_2_priorizacao_tarifa_social(params)
    analise2 = analisar_resultados(aloc2, rede2, params, "Cenário 2 - Priorização Tarifa Social")
    analises.append(analise2)
    aloc_list.append(aloc2)
    rede_list.append(rede2)

    # Executar Cenário 3
    print("\nExecutando Cenário 3: Multiobjetivo (Completo)...")
    aloc3, rede3, excedente3 = cenario_3_multiobjetivo(params)
    analise3 = analisar_resultados(aloc3, rede3, params, "Cenário 3 - Multiobjetivo Completo")
    analises.append(analise3)
    aloc_list.append(aloc3)
    rede_list.append(rede3)

    # Exportar e visualizar
    print("\nExportando resultados...")
    exportar_resultados(analises, aloc_list, params)

    print("\nGerando gráficos...")
    plotar_graficos(analises, aloc_list, rede_list, params)

    print("\nExecução concluída! Resultados salvos na pasta 'saidas/'.")

if __name__ == "__main__":
    main()
