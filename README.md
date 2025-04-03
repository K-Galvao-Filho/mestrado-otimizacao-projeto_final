# Alocação Ótima de Energia Solar em Comunidades com Geração Compartilhada

Este projeto implementa, em Python, um modelo de **Programação Linear Inteira (PLI)** para otimizar a distribuição de energia solar em comunidades com geração compartilhada, considerando critérios técnicos e de **justiça energética**.

O código está baseado no artigo submetido ao **SBPO 2025**:  
**"Alocação Ótima de Energia Solar em Comunidades com Geração Compartilhada: Uma Abordagem com Programação Linear Inteira e Critérios de Justiça Energética"**

## ✨ Objetivo
Simular e comparar três estratégias de alocação de energia solar para 10 unidades consumidoras ao longo de 24 períodos (horas), avaliando os seguintes critérios:
- **Eficiência Energética**
- **Autossuficiência**
- **Equidade na distribuição**

## 📦 Tecnologias Utilizadas
- Python 3.10+
- [PuLP](https://coin-or.github.io/pulp/) – para modelagem de Programação Linear Inteira
- NumPy e Pandas – para manipulação de dados
- Matplotlib – para visualização gráfica

## 📊 Cenários Simulados

| Cenário | Descrição |
|--------|-----------|
| **1** – Igualitário | Alocação sequencial, por ordem das unidades (método guloso) |
| **2** – PLI sem prioridade | Maximiza o número total de atendimentos sem considerar prioridade |
| **3** – PLI com prioridade | Considera pesos de prioridade e impõe uma demanda mínima para os mais vulneráveis |

## 📈 Indicadores Avaliados
- **Eficiência (%)**: energia usada ÷ energia disponível
- **Autossuficiência (%)**: energia usada ÷ demanda total
- **Equidade**: desvio padrão da energia recebida entre as unidades

## 🧪 Como Executar

1. Instale as dependências:

```bash
pip install numpy pandas matplotlib pulp
```

2. Execute o script principal:

```bash
python app.py
```

3. Os resultados serão mostrados no terminal e gráficos comparativos serão exibidos.

4. Arquivos CSV com a alocação dos cenários 1, 2 e 3 são gerados automaticamente:
- `demanda_cenario1.csv`
- `demanda_cenario2.csv`
- `demanda_cenario3.csv`
- `alocacao_cenario1.csv`
- `alocacao_cenario2.csv`
- `alocacao_cenario3.csv`

## 📁 Estrutura dos Arquivos

```
.
├── app.py                    # Código principal do projeto
├── demanda_cenario1.csv      # Demanda da alocação do cenário 1
├── demanda_cenario2.csv      # Demanda da alocação do cenário 2
├── demanda_cenario3.csv      # Demanda da alocação do cenário 3
├── alocacao_cenario1.csv     # Resultado da alocação do cenário 1
├── alocacao_cenario2.csv     # Resultado da alocação do cenário 2
├── alocacao_cenario3.csv     # Resultado da alocação do cenário 3
└── README.md                 # Este arquivo
```

## 🧠 Base Teórica

A proposta do modelo está fundamentada em literatura recente sobre justiça energética e geração distribuída no Brasil. 
A PLI é aplicada como uma forma de equilibrar **eficiência operacional** com **inclusão social**, conforme estudos de:

- Lampis et al. (2022)
- Oliveira (2022)
- Pires et al. (2022)

## 📌 Licença

Este projeto é de livre uso acadêmico e está licenciado sob os termos da Licença MIT.
