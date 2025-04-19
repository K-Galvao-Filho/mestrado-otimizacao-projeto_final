import numpy as np

def cenario_2_priorizacao_tarifa_social(params):
    """
    Cenário 2: Alocação proporcional priorizando residências da Tarifa Social.
    
    - Distribui a energia solar proporcionalmente à demanda ponderada pelo peso social.
    - Residências com Tarifa Social têm peso maior na alocação.
    - Sem considerar perdas de rede ou limites de injeção.
    """

    # Extrair parâmetros
    num_residencias = params['num_residencias']
    num_horas = params['num_horas']
    demanda_residencias = params['demanda_residencias']
    geracao_solar = params['geracao_solar']

    # Definir pesos de prioridade: 5.0 para Tarifa Social (índices pares), 1.0 para demais (índices ímpares)
    pesos_prioridade = np.array([5.0 if i % 2 == 0 else 1.0 for i in range(num_residencias)])

    # Inicializar matrizes de resultados
    aloc = np.zeros((num_residencias, num_horas))  # Energia solar alocada
    rede = np.zeros((num_residencias, num_horas))  # Energia da rede
    excedente = np.zeros(num_horas)                # Energia excedente

    # Processar hora a hora
    for t in range(num_horas):
        demanda_ponderada = demanda_residencias[:, t] * pesos_prioridade
        total_demanda_ponderada = np.sum(demanda_ponderada)

        if total_demanda_ponderada > 0:
            proporcao_ponderada = demanda_ponderada / total_demanda_ponderada
            energia_a_alocar = min(geracao_solar[t], np.sum(demanda_residencias[:, t]))
            aloc[:, t] = proporcao_ponderada * energia_a_alocar
            excedente[t] = max(0, geracao_solar[t] - np.sum(demanda_residencias[:, t]))

        # Energia que falta é puxada da rede
        rede[:, t] = np.maximum(0, demanda_residencias[:, t] - aloc[:, t])

    return aloc, rede, excedente
