
import random
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

ciudades_5 = [
    (0, 0),(2, 6),(5, 3),(6, 7),(8, 2)
]

ciudades_10 = [
    (0, 0),(2, 6),(5, 3),(6, 7),(8, 2),(1, 4),(7, 5),(3, 1),(9, 8),(4, 9)
]

ciudades_15 = [
    (0, 0),(2, 6),(5, 3),(6, 7),(8, 2),(1, 4),(7, 5),(3, 1),(9, 8),
    (4, 9),(10, 3),(2, 10),(11, 6),(6, 11),(12, 1)
]

random.seed(2011258)
r = random.random()

bajo=0.3 * r
medio=0.3 * r + 0.3
alto=0.3 * r + 0.6


T_min = 15
T_max = 30


T_bajo = T_min + bajo * (T_max - T_min)
T_medio = T_min + medio * (T_max - T_min)
T_alto = T_min + alto * (T_max - T_min)

def distancia(a, b, ciudades):
    return math.dist(ciudades[a], ciudades[b])


def distancia_ruta(ruta,ciudades):
    total = 0

    for i in range(len(ruta)):
        total += distancia(
            ruta[i],
            ruta[(i + 1) % len(ruta)],
            ciudades
        )

    return total



def vecino_2opt(ruta, rng):
    nueva = ruta.copy()

    i, j = sorted(rng.sample(range(len(ruta)), 2))
    nueva[i:j + 1] = nueva[i:j + 1][::-1]

    return nueva



def recorido_simulado(
    ruta_inicial,
    ciudades,
    temperatura_inicial,
    semilla,
    iteraciones=3000,
    temperatura_final=0.01
):
    rng = random.Random(semilla)

    actual = ruta_inicial.copy()
    costo_actual = distancia_ruta(actual, ciudades)

    mejor = actual.copy()
    mejor_costo = costo_actual

    historial = []
    inicio = time.perf_counter()

    for t in range(iteraciones):

        progreso = t / (iteraciones - 1)
        temperatura = (
            temperatura_inicial
            * (temperatura_final / temperatura_inicial)** progreso
        )

        

        siguiente = vecino_2opt(actual, rng)
        costo_siguiente = distancia_ruta(siguiente,ciudades)

        delta = costo_actual - costo_siguiente

        if delta > 0:
            aceptar = True
        else:
            probabilidad = math.exp(-max(0, -delta) / temperatura)
            aceptar = rng.random() < probabilidad

        if aceptar:
            actual = siguiente
            costo_actual = costo_siguiente

        if costo_actual < mejor_costo:
            mejor = actual.copy()
            mejor_costo = costo_actual

       
        historial.append(mejor_costo)

    tiempo = time.perf_counter() - inicio

    return mejor, mejor_costo, historial, tiempo


instancias = {
    5: ciudades_5,
    10: ciudades_10,
    15: ciudades_15
}

temperaturas = {
    "bajo": T_bajo,
    "medio": T_medio,
    "alto": T_alto
}

resultados = []

semilla = 100

for tamano, ciudades in instancias.items():

    for nivel, temperatura in temperaturas.items():

        for replica in range(1, 11):

            rng = random.Random(semilla)

            ruta_inicial = list(range(tamano))
            rng.shuffle(ruta_inicial)

            # Ejecutar Recocido Simulado
            ruta, costo, historial, tiempo = recorido_simulado(
                ruta_inicial,
                ciudades,
                temperatura_inicial=temperatura,
                semilla=semilla
            )

            # Guardar resultados
            resultados.append({
                "tamano": tamano,
                "nivel": nivel,
                "temperatura": temperatura,
                "replica": replica,
                "semilla": semilla,
                "costo": costo,
                "tiempo": tiempo
            })

            semilla += 1


df = pd.DataFrame(resultados)
print(df)
df.to_csv("resultados_doe.csv", index=False)
print("\nTotal de corridas:", len(df))
print("\nTotal de corridas:", len(df))
print("\nCorridas por tamaño:")
print(df.groupby("tamano").size())
print("\nCorridas por temperatura:")
print(df.groupby(["tamano", "nivel"]).size())



modelo = ols(
    'costo ~ C(nivel) * C(tamano)',
    data=df
).fit()

anova = sm.stats.anova_lm(modelo, typ=2)

print("\nRESULTADOS DEL ANOVA")
print(anova)

orden = ["bajo", "medio", "alto"]


#GRAFICA 1: EFECTO DE TEMPERATURA
media_temperatura = (
    df.groupby("nivel")["costo"]
    .mean()
    .reindex(orden)
)
plt.figure(figsize=(7, 5))
plt.plot(
    media_temperatura.index,
    media_temperatura.values,
    marker="o"
)
plt.title("Efecto principal de la temperatura")
plt.xlabel("Nivel de temperatura")
plt.ylabel("Costo promedio")
plt.grid(True)
plt.tight_layout()
plt.show()
#GRAFICA 2: EFECTO DEL TAMAÑO
media_tamano = df.groupby("tamano")["costo"].mean()
plt.figure(figsize=(7, 5))
plt.plot(
    media_tamano.index,
    media_tamano.values,
    marker="o"
)
plt.title("Efecto principal del tamaño")
plt.xlabel("Número de ciudades")
plt.ylabel("Costo promedio")
plt.grid(True)
plt.tight_layout()
plt.show()
#GRAFICA 3: INTERACCION TEMPERATURA-TAMAÑO
medias_interaccion = (
    df.groupby(["tamano", "nivel"])["costo"]
    .mean()
    .unstack()
    .reindex(columns=orden)
)

plt.figure(figsize=(8, 5))

for tamano in medias_interaccion.index:
    plt.plot(
        orden,
        medias_interaccion.loc[tamano],
        marker="o",
        label=f"{tamano} ciudades"
    )

plt.title("Interacción entre temperatura y tamaño")
plt.xlabel("Nivel de temperatura")
plt.ylabel("Costo promedio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import numpy as np

# Bootstrap para calcular mediana e intervalo de confianza
def bootstrap_mediana(datos, repeticiones=10000):
    rng = np.random.default_rng(42)

    medianas = []

    for i in range(repeticiones):
        muestra = rng.choice(
            datos,
            size=len(datos),
            replace=True
        )

        medianas.append(np.median(muestra))

    mediana = np.median(datos)

    limite_inferior = np.percentile(medianas, 2.5)
    limite_superior = np.percentile(medianas, 97.5)

    return mediana, limite_inferior, limite_superior


# Calcular resultados por tamaño y temperatura
resultados_bootstrap = []

for tamano in [5, 10, 15]:
    for nivel in ["bajo", "medio", "alto"]:

        datos = df[
            (df["tamano"] == tamano) &
            (df["nivel"] == nivel)
        ]["costo"].values

        mediana, inferior, superior = bootstrap_mediana(datos)

        resultados_bootstrap.append({
            "tamano": tamano,
            "nivel": nivel,
            "mediana": mediana,
            "IC_inferior": inferior,
            "IC_superior": superior
        })


df_bootstrap = pd.DataFrame(resultados_bootstrap)

print("\nRESULTADOS DEL BOOTSTRAP")
print(df_bootstrap.round(4))

df_bootstrap.to_csv(
    "resultados_bootstrap.csv",
    index=False
)

