# Alocação Ótima de Energia Solar em Comunidades com Geração Compartilhada

Este projeto implementa, em Python, um modelo de **Programação Linear Inteira (PLI)** para otimizar a distribuição de energia solar fotovoltaica em comunidades com geração compartilhada. A proposta integra **critérios técnicos** e **critérios de justiça energética**, visando maximizar o aproveitamento local da energia renovável e promover maior equidade entre os consumidores.

O sistema baseia-se no artigo submetido ao **SBPO 2025**:
**"Alocação Ótima de Energia Solar em Comunidades com Geração Compartilhada via Programação Linear Inteira:
Uma Abordagem Integrada com Critérios de Justiça Energética"**.

## 🎯 Objetivo

Comparar diferentes estratégias de alocação de energia solar ao longo de 24 períodos (horas) para 10 unidades consumidoras, por meio de simulações computacionais baseadas em dados sintéticos. As abordagens analisadas têm como finalidade:

- Maximizar a eficiência energética da comunidade;
- Reduzir a dependência da rede convencional;
- Promover justiça distributiva com base em prioridades socioeconômicas.

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **PuLP** – para modelagem e resolução dos problemas de otimização
- **NumPy** e **Pandas** – para geração e manipulação de dados
- **Matplotlib** – para visualizações gráficas
- **Solver CBC** (Coin-or Branch and Cut) – utilizado via PuLP

## 🧪 Cenários Simulados

Três cenários distintos foram modelados e comparados:

| Cenário | Descrição |
|--------|-----------|
| **1. Alocação sequencial (igualitária)** | Energia distribuída por ordem das unidades, sem otimização |
| **2. PLI sem pesos sociais** | Maximiza o número total de atendimentos, sem considerar vulnerabilidades |
| **3. PLI com pesos e meta mínima** | Integra pesos socioeconômicos (prioridade) e exige demanda mínima atendida para os mais vulneráveis |

## 📊 Indicadores Avaliados

A análise de desempenho considera os seguintes indicadores:

- **Eficiência energética (%)**: razão entre a energia alocada e a energia solar disponível.
- **Autossuficiência (%)**: porcentagem da demanda total atendida.
- **Equidade (Desvio padrão)**: quanto menor o desvio, mais justa é a alocação entre consumidores.

Esses dados são calculados por meio da função `analisar()` definida no script principal.

## 📈 Resultados Visuais

O projeto gera automaticamente dois gráficos:

- **Distribuição de Energia por Unidade**: compara a energia total recebida por cada unidade nos três cenários.
- **Indicadores Comparativos**: gráfico de barras com eficiência, autossuficiência e equidade.

As visualizações seguem boas práticas de reprodutibilidade científica, sendo compatíveis com exportação para LaTeX via `pgfplots`.

## 📂 Estrutura dos Arquivos

```
.
├── app.py                    # Código principal com simulações e análise
├── saidas/
│   ├── metricas.csv              # Métricas de desempenho dos três cenários
│   ├── demanda_cenario1.csv      # Demanda efetiva por unidade – cenário 1
│   ├── demanda_cenario2.csv      # Demanda efetiva – cenário 2
│   ├── demanda_cenario3.csv      # Demanda efetiva – cenário 3
│   ├── alocacao_cenario1.csv     # Alocação binária – cenário 1
│   ├── alocacao_cenario2.csv     # Alocação binária – cenário 2
│   ├── alocacao_cenario3.csv     # Alocação binária – cenário 3
└── README.md
```

## 🚀 Como Executar

1. Instale as dependências:

```bash
pip install numpy pandas matplotlib pulp
```

2. Execute o script principal:

```bash
python app.py
```

3. Os gráficos serão exibidos automaticamente e os resultados serão salvos na pasta `saidas/`.

## 🧠 Fundamentação Científica

O modelo e os experimentos estão baseados em literatura nacional recente sobre justiça energética, geração distribuída e modelagem por PLI. Referências principais:

- Lampis et al. (2022) – Justiça energética e desigualdades na GD
- Oliveira (2022) – PLI para confiabilidade em microrredes
- Pires et al. (2022) – Otimização multiobjetivo na GD híbrida
- Ribeiro Morais et al. (2024) – Marco Legal da GD (Lei 14.300/2022)

## 📝 Licença

Este projeto é de uso acadêmico e está licenciado sob os termos da Licença MIT.
