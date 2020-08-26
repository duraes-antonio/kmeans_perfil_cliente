import collections
import io
from typing import List, Iterable

import matplotlib.pyplot as pyplt
import pandas
import requests
from sklearn.cluster import KMeans


def calc_inercia_por_cluster(valores: Iterable[Iterable[float]], k_min=1, k_max=10) -> List[float]:
    """
    Calcula a inércia gerada para cada número de cluster (entre k_min e k_max

    :param valores: Lista com as colunas de valores reais
    :param k_min: Mínimo de clusters a ser simulados
    :param k_max: Máximo de clusters a ser simulados
    :return: Lista com valores de inércia
    """
    return [
        KMeans(i, max_iter=500, random_state=42).fit(valores).inertia_
        for i in range(k_min, k_max + 1)
    ]


def plotar_inercia_cluster(inercia_valores: List[float], k_min=1):
    pyplt.figure(figsize=(8, 4))
    pyplt.plot(range(k_min, k_min + len(inercia_valores)), inercia_valores)
    pyplt.xlabel('Quantidade de clusters')
    pyplt.ylabel('Valor de inércia')
    pyplt.grid(True)
    pyplt.show()


def calc_qtd_cluster(inercias: List[float], variancia_aceita=25) -> int:
    """
    Calcula a qtd. ideal de cluster c/ base em uma variação aceita entre
    uma inércia (i) e sua próxima (i+1)

    :param inercias: Lista de valores de inércia
    :param variancia_aceita: Threshold de variação entre as inércias
    :return: Número adequado de clusters se for encontrado, senão, len(inercias) + 1
    """
    variacias = [inercias[i] / inercias[i + 1] for i in range(len(inercias) - 1)]

    # Percorra os índices das amplitudes das inércias
    for i in range(len(variacias) - 1):

        # Se a diferença entre uma amplitude e a sua próxima for menor
        # ou igual ao valor aceito, retorne o número de clusters encontrado
        if (abs(variacias[i] - variacias[i + 1]) * 100 <= variancia_aceita):
            # Incremente 1, pois i começa em 0, e mais 1, pois a variação
            # é calcula entre pares de valores
            return i + 2

    return len(inercias) + 1


def download_arq(path: str, encoding='utf8') -> object:
    r: requests.Response = requests.get(path)
    return io.StringIO(r.content.decode(encoding))


def main():
    # Abra o arquivo e defina o separador de colunas
    path_arq_remoto = 'https://raw.githubusercontent.com/duraes-antonio/kmeans_perfil_cliente/master/data/customer_age_book_price.txt'
    data_frame = pandas.read_csv(download_arq(path_arq_remoto), sep=' ')
    df_valores = data_frame.values

    # Com base na inércia calculada pelo kmeans, identifique a qtd. de clusters
    qtd_cluster = calc_qtd_cluster(calc_inercia_por_cluster(df_valores))

    # Invoque o KMeans passando a qtd. de cluster e chame o método p/ rotular os valores
    kmeans = KMeans(n_clusters=qtd_cluster, max_iter=500, random_state=42)
    predic = kmeans.fit_predict(df_valores)

    cores = ['orange', 'red', 'purple', 'blue']
    labels = ['Usual', 'Ideal', 'Comum - Mais velho', 'Comum - Jovem']

    # Ordene cada rótulo pela quantidade de elementos. Isso para manter a
    # cor e label correto independente de qual rótulo o KMeans atribua aos
    # dados, uma vez que o rótulo pode variar a cada execução
    cont_pontos = {k: v for k, v in sorted(collections.Counter(predic).items(), key=lambda item: item[1])}

    pyplt.figure(figsize=(24, 16))

    for i, v in enumerate(cont_pontos):
        pyplt.scatter(
            df_valores[predic == v, 0], df_valores[predic == v, 1],
            s=100, c=cores[i], label=labels[i], alpha=.5
        )

    pyplt.title('Clusters Idade / Consumo', fontsize=14)
    pyplt.xlabel('Idade')
    pyplt.ylabel('Consumo (R$)')
    pyplt.legend()
    pyplt.show()
    return 0


main()
