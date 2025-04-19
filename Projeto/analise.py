import numpy as np

def analisar_resultados(aloc, rede, params, nome):
    """Calcula métricas de desempenho."""
    demanda_residencias = params['demanda_residencias']
    geracao_solar = params['geracao_solar']
    tarifas_residencias = params['tarifas_residencias']
    num_residencias = params['num_residencias']
    num_horas = params['num_horas']
    
    energia_total = aloc.sum()
    custo_total = sum(tarifas_residencias[i][t] * rede[i][t] for i in range(num_residencias) for t in range(num_horas))
    eficiencia = energia_total / sum(geracao_solar) * 100
    autossuf = energia_total / demanda_residencias.sum() * 100
    eq_std = aloc.sum(axis=1).std()
    
    print(f"\nEnergia total alocada - {nome}: {energia_total:.2f} kWh")
    print(f"Custo total da rede - {nome}: R${custo_total:.2f}")
    print(f"Eficiência: {eficiencia:.2f}%")
    print(f"Autossuficiência: {autossuf:.2f}%")
    print(f"Equidade (Desvio Padrão): {eq_std:.2f}")
    print(f"Uso da rede: {rede.sum():.2f} kWh")
    
    return {
        "cenário": nome,
        "eficiência (%)": eficiencia,
        "autossuficiência (%)": autossuf,
        "desvio padrão": eq_std,
        "rede (kWh)": rede.sum(),
        "custo (R$)": custo_total,
        "energia_unidade": aloc.sum(axis=1)
    }