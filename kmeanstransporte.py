# Agrupamiento con K-means - Transporte Manizales
# Actividad 4 - Metodos No Supervisados
# Corporacion Universitaria Iberoamericana

# Este programa agrupa los viajes del sistema de transporte
# usando K-means, que es un metodo no supervisado.
# A diferencia del arbol de decision, aqui el modelo no sabe
# de antemano si un viaje es rapido o demorado.
# El solo encuentra grupos de viajes parecidos entre si.

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score


print("=" * 50)
print("  SISTEMA DE TRANSPORTE - MANIZALES")
print("  Modelo: Agrupamiento K-means")
print("=" * 50)

datos = pd.read_csv("dataset_viajes.csv")
print(f"\nDatos cargados: {len(datos)} registros")



le_dia     = LabelEncoder()
le_origen  = LabelEncoder()
le_destino = LabelEncoder()

datos["dia_num"]     = le_dia.fit_transform(datos["dia_semana"])
datos["origen_num"]  = le_origen.fit_transform(datos["origen"])
datos["destino_num"] = le_destino.fit_transform(datos["destino"])


X = datos[["hora", "dia_num", "origen_num", "destino_num", "pasajeros"]]


scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

print("Datos preparados y escalados correctamente.")


print("\n" + "-" * 50)
print("Evaluando numero de grupos:")
print(f"  {'Grupos':<10} {'Silhouette Score'}")
print("  " + "-" * 28)

for k in [2, 3, 4]:
    modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas_temp = modelo_temp.fit_predict(X_escalado)
    score = silhouette_score(X_escalado, etiquetas_temp)
    print(f"  {k:<10} {round(score, 4)}")


print("\n" + "-" * 50)
print("Entrenando modelo con 3 grupos...")

modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
datos["grupo"] = modelo.fit_predict(X_escalado)

print("Entrenamiento completado.")


print("\n" + "-" * 50)
print("Caracteristicas de cada grupo:\n")

nombres_grupos = {}

for g in sorted(datos["grupo"].unique()):
    subconjunto = datos[datos["grupo"] == g]
    hora_prom   = round(subconjunto["hora"].mean(), 1)
    pasaj_prom  = round(subconjunto["pasajeros"].mean(), 1)
    tiempo_prom = round(subconjunto["tiempo_real"].mean(), 1)
    cantidad    = len(subconjunto)


    if pasaj_prom > 110:
        nombre = "Viajes de hora pico"
    elif pasaj_prom > 60:
        nombre = "Viajes intermedios"
    else:
        nombre = "Viajes tranquilos"

    nombres_grupos[g] = nombre

    print(f"  Grupo {g} - {nombre}")
    print(f"  Cantidad de viajes : {cantidad}")
    print(f"  Hora promedio      : {hora_prom}h")
    print(f"  Pasajeros promedio : {pasaj_prom}")
    print(f"  Tiempo promedio    : {tiempo_prom} minutos")


    estado_mas_comun = subconjunto["estado"].value_counts().idxmax()
    print(f"  Estado mas comun   : {estado_mas_comun}")
    print()

print("-" * 50)
print("Comparacion: grupo asignado vs estado real\n")
print(f"  {'Grupo':<22} {'Demorado':>10} {'Normal':>10} {'Rapido':>10}")
print("  " + "-" * 54)

for g in sorted(datos["grupo"].unique()):
    subconjunto = datos[datos["grupo"] == g]
    conteo = subconjunto["estado"].value_counts()
    dem = conteo.get("Demorado", 0)
    nor = conteo.get("Normal",   0)
    rap = conteo.get("Rapido",   0)
    print(f"  {nombres_grupos[g]:<22} {dem:>10} {nor:>10} {rap:>10}")


print("\n" + "-" * 50)
print("Prediccion para viajes nuevos:\n")

nuevos = [
    {"hora": 7,  "dia": "Lunes",   "origen": "Cable Aereo", "destino": "Universidad", "pasajeros": 135},
    {"hora": 11, "dia": "Domingo", "origen": "Centro",      "destino": "Palermo",     "pasajeros": 28},
    {"hora": 17, "dia": "Viernes", "origen": "Palermo",     "destino": "La Enea",     "pasajeros": 150},
]

for nv in nuevos:
    dia_n     = le_dia.transform([nv["dia"]])[0]     if nv["dia"]     in le_dia.classes_     else 0
    origen_n  = le_origen.transform([nv["origen"]])[0]  if nv["origen"]  in le_origen.classes_  else 0
    destino_n = le_destino.transform([nv["destino"]])[0] if nv["destino"] in le_destino.classes_ else 0

    entrada = [[nv["hora"], dia_n, origen_n, destino_n, nv["pasajeros"]]]
    entrada_esc = scaler.transform(entrada)
    grupo_pred = modelo.predict(entrada_esc)[0]

    print(f"  {nv['hora']}h {nv['dia']:<10} | {nv['origen']:<12} -> {nv['destino']:<12} | {nv['pasajeros']} pasaj. => {nombres_grupos[grupo_pred]}")

print("\n" + "=" * 50)
