import numpy as np

def cenario_1_proporcional_simples(params):
    """
    Cenário 1: Alocação proporcional simples da geração solar.
    
    - Distribui a energia solar proporcionalmente à demanda de cada residência.
    - Sem priorização, sem considerar eficiência de rede ou limite de injeção.
    - Demanda não atendida é complementada pela rede elétrica.
    """

    # Extrair parâmetros
    num_residencias = params['num_residencias']
    num_horas = params['num_horas']
    demanda_residencias = params['demanda_residencias']
    geracao_solar = params['geracao_solar']

    # Inicializar matrizes de resultados
    aloc = np.zeros((num_residencias, num_horas))  # Energia solar alocada
    rede = np.zeros((num_residencias, num_horas))  # Energia da rede
    excedente = np.zeros(num_horas)                # Energia excedente

    # Processar hora a hora
    for t in range(num_horas):
        total_demanda = np.sum(demanda_residencias[:, t])

        if total_demanda > 0:
            proporcao = demanda_residencias[:, t] / total_demanda
            energia_a_alocar = min(geracao_solar[t], total_demanda)
            aloc[:, t] = proporcao * energia_a_alocar
            excedente[t] = max(0, geracao_solar[t] - total_demanda)

        # Energia que falta é puxada da rede
        rede[:, t] = np.maximum(0, demanda_residencias[:, t] - aloc[:, t])

    return aloc, rede, excedente
