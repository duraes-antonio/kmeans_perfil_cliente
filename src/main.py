import collections
from typing import List, Iterable

import matplotlib.pyplot as pyplt
import pandas
from sklearn.cluster import KMeans


def calc_inercia_por_cluster(valores: Iterable[Iterable], k_min=1, k_max=10) -> List[float]:
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
    variacias = [inercias[i] / inercias[i + 1] for i in range(len(inercias) - 1)]

    # Percorra os índices das amplitudes das inércias
    for i in range(len(variacias) - 1):

        # Se a diferença entre uma amplitude e a sua próxima for menor
        # ou igual ao valor aceito, retorne o número de clusters encontrado
        if (abs(variacias[i] - variacias[i + 1]) * 100 <= variancia_aceita):
            # Incremente 1, pois i começa em 0, e mais 1, pois a variação
            # é calcula entre pares de valores
            return i + 2


def main():
    data_frame = pandas.read_csv('../data/customer_age_book_price.txt', sep=' ')
    df_valores = data_frame.values
    qtd_cluster = calc_qtd_cluster(calc_inercia_por_cluster(df_valores))

    kmeans = KMeans(n_clusters=qtd_cluster, max_iter=500, random_state=42)
    predic = kmeans.fit_predict(df_valores)

    cores = ['orange', 'red', 'purple', 'blue']
    labels = ['Usual', 'Ideal', 'Comum - Mais velho', 'Comum - Jovem']
    cont_pontos = {k: v for k, v in sorted(collections.Counter(predic).items(), key=lambda item: item[1])}

    pyplt.figure(figsize=(24, 16))

    for i, v in enumerate(cont_pontos):
        pyplt.scatter(
            df_valores[predic == v, 0], df_valores[predic == v, 1], s=100,
            c=cores[i], label=labels[i], alpha=.5
        )

    pyplt.title('Clusters Idade / Consumo', fontsize=14)
    pyplt.xlabel('Idade')
    pyplt.ylabel('Consumo (R$)')
    pyplt.legend()
    pyplt.show()
    return 0


main()
