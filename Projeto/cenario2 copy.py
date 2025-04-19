import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpContinuous, PULP_CBC_CMD

def cenario_2_mono_objetivo(params):
    """Maximiza alocação solar priorizando Tarifa Social, conforme SCEE."""
    num_residencias = params['num_residencias']
    num_horas = params['num_horas']
    demanda_residencias = params['demanda_residencias']
    geracao_solar = params['geracao_solar']
    eficiencia_rede = params['eficiencia_rede']
    pesos_prioridade_c2 = params['pesos_prioridade_c2']
    demanda_base_ajustada = params['demanda_base_ajustada']
    demandas_minimas_residencias = params['demandas_minimas_residencias']
    limite_fluxo_hora = params['limite_fluxo_hora']
    limite_injecao_rede = params['limite_injecao_rede']

    modelo = LpProblem("Cenario_2", LpMaximize)

    x = LpVariable.dicts("x", ((i, t) for i in range(num_residencias) for t in range(num_horas)), 0, 1, LpContinuous)
    y = LpVariable.dicts("y", ((i, t) for i in range(num_residencias) for t in range(num_horas)), cat=LpBinary)
    u = LpVariable.dicts("u", ((i, t) for i in range(num_residencias) for t in range(num_horas)), lowBound=0, cat=LpContinuous)
    z = LpVariable.dicts("z", (t for t in range(num_horas)), lowBound=0, cat=LpContinuous)

    modelo += lpSum(pesos_prioridade_c2[i] * x[i, t] * demanda_residencias[i][t] for i in range(num_residencias) for t in range(num_horas))

    for t in range(num_horas):
        modelo += lpSum(x[i, t] * demanda_residencias[i][t] for i in range(num_residencias)) <= eficiencia_rede[t] * geracao_solar[t]
        modelo += lpSum(x[i, t] * demanda_residencias[i][t] for i in range(num_residencias)) <= limite_fluxo_hora
        modelo += lpSum(x[i, t] * demanda_residencias[i][t] for i in range(num_residencias)) + z[t] <= eficiencia_rede[t] * geracao_solar[t]
        modelo += z[t] <= limite_injecao_rede[t]

    for i in range(num_residencias):
        modelo += lpSum(x[i, t] * demanda_residencias[i][t] for t in range(num_horas)) >= demanda_base_ajustada
        modelo += lpSum(x[i, t] * demanda_residencias[i][t] for t in range(num_horas)) >= demandas_minimas_residencias[i]
        for t in range(num_horas):
            modelo += x[i, t] * demanda_residencias[i][t] + u[i, t] >= demanda_residencias[i][t]
            modelo += x[i, t] <= y[i, t]

    modelo.solve(PULP_CBC_CMD(msg=1, timeLimit=300))  # Ativar mensagens do solver para depuração

    print(f"Status do solver (Cenário 2): {modelo.status}")  # 1 = ótimo, 0 = não resolvido, -1 = inviável, etc.

    if modelo.status != 1:
        print("Erro: Solver não encontrou solução ótima no Cenário 2")
        raise ValueError("Solver falhou para o Cenário 2")

    aloc = np.zeros((num_residencias, num_horas))
    rede = np.zeros((num_residencias, num_horas))
    excedente = np.zeros(num_horas)

    for i in range(num_residencias):
        for t in range(num_horas):
            aloc[i, t] = max(0, x[i, t].varValue or 0) * demanda_residencias[i][t]
            rede[i, t] = max(0, u[i, t].varValue or 0)  # Corrigido: usa apenas u[i, t]
    for t in range(num_horas):
        excedente[t] = max(0, z[t].varValue or 0)

    return aloc, rede, excedente