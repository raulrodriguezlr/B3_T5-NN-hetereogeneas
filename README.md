# Práctica B3–T5 · Redes neuronales con entradas heterogéneas

**Máster — Predicción de series temporales con Deep Learning**

**Grupo:** Alonso Díaz · Raúl Rodríguez · Piettro Enrico
**Entrega:** 13 de junio (aula virtual) — un único *notebook* (`.ipynb`)

---

## 1. Objetivo de la práctica

Construir y **optimizar un modelo neuronal con entradas heterogéneas** para un
problema de **predicción de series temporales de ventas**. "Entradas
heterogéneas" significa que la red combina, en una misma arquitectura, señales
de naturaleza distinta:

- una **variable endógena** (la propia serie de ventas, su pasado reciente),
- **variables exógenas numéricas** (p. ej. distancia a la competencia, nº de clientes),
- **variables categóricas** que entran a la red mediante **embeddings**
  (tipo de tienda, día de la semana, mes, promoción…).

El problema procede de la competición de Kaggle **Rossmann Store Sales**
(predicción de ventas diarias en una cadena de ~1.100 tiendas).

## 2. Criterios de evaluación (según enunciado)

| Peso | Criterio |
|------|----------|
| 50 % | **Resultados**: coeficiente de determinación **R²** en test de las **10 tiendas**. |
| 50 % | **Análisis y justificación**: calidad del análisis, coherencia de la estrategia y reflexión crítica. |

> El entregable debe contener: nombres del grupo, código del modelo,
> **justificación de arquitectura y preprocesados**, **resumen de métricas** de
> distintas arquitecturas y una **reflexión final**.

El PDF original del enunciado está en [`docs/enunciado_practica.pdf`](docs/enunciado_practica.pdf).

## 3. Datos

Los datos viven en [`data/`](data/) (ver el diccionario completo en
[`data/Data.md`](data/Data.md)). Resumen:

| Fichero | Filas | Contenido |
|---------|-------|-----------|
| `train.zip` → `train.csv` | ~1.000.000 | Ventas diarias por tienda. **2013-01-01 → 2015-07-17**, 1.115 tiendas. |
| `store.csv` | 1.115 | Metadatos de cada tienda (tipo, surtido, competencia, promociones). |
| `test.csv` | 1.115 | Una fila por tienda para el día **2015-07-31** (sin la columna `Sales`). |
| `submission.csv` | 1.115 | Formato de entrega de ejemplo (`Id`, `Sales`). |

> ⚠️ `train.csv` se distribuye comprimido (`train.zip`, ~11 MB) porque
> descomprimido supera el límite cómodo de GitHub.

**Observación importante sobre el horizonte de predicción:** el entrenamiento
termina el **2015-07-17** y el test es el **2015-07-31**. Hay un hueco de ~2
semanas sin datos entre ambos, por lo que la predicción del día de test no es
"a un día" sino un **forecast a ~2 semanas vista** (multi-step). Esto condiciona
la estrategia (ver [`docs/ESTRATEGIA.md`](docs/ESTRATEGIA.md)).

## 4. Notebooks base (punto de partida)

El profesor nos da tres notebooks de referencia (autor: Manuel
Sánchez-Montañés) que construyen el modelo de forma **incremental** sobre un
dataset sencillo de ejemplo (`datos_pasajeros.csv`). Están en
[`notebooks_base/`](notebooks_base/) y conviene leerlos **en orden**:

| # | Notebook | Idea clave | Qué aporta hacia la práctica |
|---|----------|-----------|------------------------------|
| 01 | `01_usando_solo_endogena.ipynb` | Modelo *many-to-one* usando **solo el pasado de la serie**. GRU(2) + Dense(1). | Enventanado (`enventanar`), train/test temporal, predicción a 1 día y autoregresiva. |
| 02 | `02_endogena_y_exogenas.ipynb` | Añade **exógenas** (mes, semana, día de la semana) con **one-hot**. LSTM(5) + Dense(1). | Cómo mezclar endógena + exógenas conocidas de antemano; modelos persistentes de referencia. |
| 03 | `03_con_embeddings.ipynb` | **Entradas heterogéneas con `Embedding`** (API funcional de Keras): un `Input` por variable, embeddings para categóricas, `concatenate` + LSTM. | **Es la plantilla directa de la práctica.** Embeddings, multi-step, visualización de embeddings y bloque de "producción". |

> Los notebooks base usan una librería auxiliar `my_utils_series_temporales`
> (`int2dummy`, `enventanar`, `info_enventanado`, `NAN`) que **se descarga de
> Google Drive** al ejecutar en Colab (`COLAB = True`). No está en el repo
> porque no tenemos el fichero fuente; al ejecutar en Colab se descarga sola.
> Ver [`docs/ESTRATEGIA.md`](docs/ESTRATEGIA.md) para qué hace cada función.

## 5. Estructura del repositorio

```
.
├── README.md                  ← este documento
├── requirements.txt           ← dependencias de Python
├── data/                      ← datos de Rossmann (train/store/test/submission)
│   └── Data.md                ← diccionario de datos
├── docs/
│   ├── enunciado_practica.pdf ← enunciado original
│   └── ESTRATEGIA.md          ← plan de trabajo y decisiones de diseño
└── notebooks_base/            ← los 3 notebooks de referencia + datos_pasajeros.csv
```

Hay **dos notebooks entregables** sobre los datos de Rossmann (ambos siguen la
plantilla del notebook 03, entradas heterogéneas con embeddings):

| Notebook | Para qué |
|----------|----------|
| [`notebook_entregable_rossmann.ipynb`](notebook_entregable_rossmann.ipynb) | **Versión didáctica**: esquema incremental A (solo endógena) → B (+exógenas) → C (+embeddings), con comparativa. Ideal para explicar el porqué de cada paso. |
| [`notebook_entregable_rossmann_afinado.ipynb`](notebook_entregable_rossmann_afinado.ipynb) | **Versión afinada** (mejor R²): canales exógenos dentro de la LSTM, *feature engineering* (lags semanales, medias de ventana), LSTM apilada, búsqueda de hiperparámetros, RMSPE y forecast multi-step con calendario real. |

La hoja de ruta y las decisiones de diseño están en
[`docs/ESTRATEGIA.md`](docs/ESTRATEGIA.md). La **explicación detallada celda por
celda** del notebook afinado (con resultados de una ejecución real) está en
[`docs/EXPLICACION_NOTEBOOK.md`](docs/EXPLICACION_NOTEBOOK.md) y en
[`docs/EXPLICACION_NOTEBOOK.pdf`](docs/EXPLICACION_NOTEBOOK.pdf).

> **"Las 10 tiendas" = las 10 de mayor volumen de ventas.** El R² se reporta de
> forma global y restringido a esas 10 tiendas. Como `test.csv` (2015-07-31) no
> trae `Sales`, el R² lo medimos sobre un **hold-out temporal interno** (últimas
> semanas de train) y además generamos la predicción de 2015-07-31 para entregar.

## 6. Cómo empezar

```bash
pip install -r requirements.txt
jupyter notebook            # o abrir los notebooks en Colab
```

Para descomprimir los datos de entrenamiento localmente:

```bash
cd data && unzip -o train.zip
```

## 7. Estado actual y próximos pasos

- [x] Repositorio organizado y documentado.
- [x] Diccionario de datos y análisis del enunciado.
- [x] Plan de estrategia (`docs/ESTRATEGIA.md`).
- [x] Decisión sobre "las 10 tiendas": las 10 de mayor volumen de ventas.
- [x] **Notebook entregable didáctico** (`notebook_entregable_rossmann.ipynb`): EDA, baselines, modelos A/B/C, comparativa, embeddings, submission y reflexión.
- [x] **Notebook entregable afinado** (`notebook_entregable_rossmann_afinado.ipynb`): canales exógenos en la LSTM, feature engineering, búsqueda de hiperparámetros, RMSPE y forecast multi-step con calendario real.
- [ ] Ejecutar en Colab/GPU, ajustar hiperparámetros y rellenar la reflexión con los R² obtenidos.
- [ ] Revisar en grupo y pulir la justificación/análisis.
