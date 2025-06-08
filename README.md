# Algoritmo Genético para Otimização da Trajetória de Drone (Problema do Caixeiro Viajante 3D)

Este repositório contém uma implementação em Python de um Algoritmo Genético (AG) projetado para resolver uma variante tridimensional (3D) do Problema do Caixeiro Viajante (PCV). O objetivo é encontrar uma trajetória subótima para um drone que precisa visitar um conjunto de pontos pré-definidos em um espaço 3D e retornar à sua origem, minimizando a distância euclidiana total percorrida.

Este projeto foi desenvolvido como parte de um trabalho acadêmico para a disciplina de Inteligência Artificial.

## Visão Geral do Problema

O Problema do Caixeiro Viajante é um desafio clássico de otimização NP-difícil. A sua extensão para três dimensões é particularmente relevante para aplicações modernas, como o planejamento de rotas para Veículos Aéreos Não Tripulados (VANTs) ou drones. A otimização dessas trajetórias pode levar a economias significativas de energia, tempo e custos operacionais.

## Funcionalidades Implementadas

*   **Carregamento de Dados:** Os pontos são carregados de um arquivo CSV (`CaixeiroGruposGA.csv`), onde cada linha define as coordenadas (x, y, z) de um ponto.
*   **Seleção de Pontos:**
    *   O último ponto no CSV é designado como a origem.
    *   O algoritmo permite configurar um número alvo de pontos visitáveis (entre 30 e 60, conforme especificado nos requisitos originais do trabalho).
    *   Se o número de pontos disponíveis no CSV (excluindo a origem) exceder o máximo alvo, uma amostra aleatória é selecionada para a execução.
*   **Representação do Indivíduo:** Cada indivíduo (rota potencial) na população do AG é representado como uma permutação dos índices dos pontos a serem visitados.
*   **Função de Aptidão (Fitness):** A aptidão de um indivíduo é inversamente proporcional à distância euclidiana 3D total da rota (incluindo a ida da origem ao primeiro ponto e o retorno do último ponto à origem).
*   **Operadores Genéticos:**
    *   **Seleção:** Seleção por torneio.
    *   **Recombinação (Crossover):** Operador de recombinação de permutação de dois pontos personalizado, garantindo que os filhos gerados não contenham pontos repetidos.
    *   **Mutação:** Mutação por troca (swap), onde duas posições aleatórias na sequência de um indivíduo têm seus genes trocados.
    *   **Elitismo:** Um número configurável dos melhores indivíduos da geração atual é diretamente transferido para a próxima geração.
*   **Critérios de Parada:** O AG para quando:
    *   Atinge um número máximo de gerações pré-definido.
    *   OU, não há melhoria significativa no custo (distância) da melhor solução por um número especificado de gerações consecutivas.
*   **Análise Estatística:**
    *   O script pode executar o AG múltiplas vezes para coletar dados sobre o desempenho.
    *   Calcula e exibe a "moda de gerações" em que o algoritmo tende a atingir um critério de parada (estagnação ou máximo de gerações).
    *   Fornece estatísticas (média, mínimo, máximo, desvio padrão) sobre as melhores distâncias encontradas nas múltiplas execuções.
*   **Visualização:**
    *   Gera gráficos da melhor rota 3D encontrada, mostrando todos os pontos, a origem, os pontos visitados na ordem e o caminho da rota.
    *   Gera gráficos de convergência, mostrando a evolução da melhor distância (custo) ao longo das gerações.
    *   As imagens geradas podem ser salvas automaticamente em um diretório configurável (padrão: `ga_tsp_results`).

## Estrutura do Código

O código está contido no arquivo `main.py` (ou nome similar que você usou) e é organizado nas seguintes seções principais:

1.  **Parâmetros de Configuração:** Constantes globais para controlar o comportamento do AG, seleção de pontos, análise estatística e saída de imagens.
2.  **Carregamento e Preparação de Dados:** Função `load_points_numpy` para ler o CSV e selecionar os pontos.
3.  **Funções Centrais do AG:** Funções para cálculo de distância, criação de indivíduos, inicialização da população e cálculo de fitness.
4.  **Operadores Genéticos:** Funções para seleção por torneio, crossover e mutação.
5.  **Loop Principal do AG:** Função `genetic_algorithm` que orquestra o processo evolutivo.
6.  **Visualização:** Funções `plot_route` e `plot_fitness_history` para gerar e salvar os gráficos.
7.  **Execução Principal (`if __name__ == "__main__":`)**:
    *   Gerencia as múltiplas execuções para análise estatística.
    *   Chama as funções de carregamento, o AG e as funções de plotagem/salvamento.
    *   Imprime o resumo da análise estatística e informações de configuração para relatório.

## Bibliotecas Necessárias

*   **NumPy:** Para cálculos numéricos eficientes, especialmente operações vetoriais e manipulação de arrays.
    ```bash
    pip install numpy
    ```
*   **Matplotlib:** Para a geração dos gráficos 2D e 3D.
    ```bash
    pip install matplotlib
    ```
*   Nenhuma outra biblioteca externa é estritamente necessária, pois o código utiliza módulos padrão do Python como `csv`, `random`, `collections` e `os`.

## Como Executar

1.  **Pré-requisitos:** Certifique-se de ter o Python 3 instalado e as bibliotecas NumPy e Matplotlib.
2.  **Arquivo de Dados:** Coloque o arquivo de dados `CaixeiroGruposGA.csv` no mesmo diretório que o script Python, ou ajuste o caminho no script.
3.  **Execução:** Abra um terminal ou prompt de comando, navegue até o diretório do projeto e execute o script:
    ```bash
    python main.py
    ```

4.  **Configuração:** Você pode ajustar os parâmetros no início do script, como `POPULATION_SIZE`, `MAX_GENERATIONS`, `NUM_RUNS_FOR_STATS`, `OUTPUT_IMAGE_DIR`, etc., para experimentar diferentes configurações.

## Resultados Esperados

Ao executar o script:

*   O console exibirá informações sobre o progresso de cada execução do AG, incluindo a melhor distância encontrada em cada geração (com menor frequência para execuções longas).
*   Ao final de todas as execuções estatísticas, um resumo será impresso, incluindo:
    *   A moda, média e mediana das gerações em que o AG parou.
    *   Estatísticas sobre as melhores distâncias totais encontradas.
    *   Um resumo da configuração usada, útil para relatórios.
*   Se `SAVE_PLOTS` for `True` (padrão), as imagens dos gráficos de rota e convergência para cada execução serão salvas no diretório especificado por `OUTPUT_IMAGE_DIR` (padrão: `ga_tsp_results`).

## Exemplo de Estrutura do Arquivo CSV (`CaixeiroGruposGA.csv`)

O arquivo CSV deve conter quatro colunas, representando as coordenadas x, y, z e um identificador de grupo (este último não é utilizado diretamente para o cálculo da rota no AG atual, mas está presente no formato de dados original).

```csv
1.465308465661800774e+01,4.540125019840555609e+01,2.606186412151147991e+00,3.000000000000000000e+00
3.531770612008283194e+01,5.164142354372653898e+01,-2.147781318796592842e+00,3.000000000000000000e+00
...
0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
```
**Nota:** O último ponto no arquivo CSV é tratado como o ponto de origem e destino do drone.

## Licença

Este projeto é disponibilizado sob a Licença MIT.
