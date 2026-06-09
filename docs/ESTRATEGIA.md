# Estrategia y plan de trabajo

Documento vivo: aquí dejamos las decisiones de diseño, el mapeo de los
notebooks base al problema de Rossmann y las dudas abiertas.

## 1. De los notebooks base a la práctica

Los tres notebooks base resuelven el **mismo problema** (predecir una serie)
con complejidad creciente. La práctica nos pide llegar al nivel del **notebook
03 (embeddings / entradas heterogéneas)** pero aplicado a Rossmann.

```
01 endógena ──► 02 + exógenas one-hot ──► 03 + embeddings (entradas heterogéneas)
                                                  │
                                                  └──► PLANTILLA de la práctica
```

### Qué hace la librería `my_utils_series_temporales`

(Se descarga de Drive en Colab; conviene entenderla para reimplementarla si
trabajamos en local.)

- **`int2dummy(arr, min, max)`** — one-hot de un array de enteros, con categorías
  desde `min` hasta `max` (ambos inclusive). Devuelve forma `(N, max-min+1)`.
- **`enventanar(series, target, se_saben_antes, W_in)`** — convierte una lista de
  series 1-D en ventanas deslizantes de tamaño `W_in` (*lookback*). Devuelve
  `X` de forma `(N, W_in, n_series)` e `y` de forma `(N,)` con el valor de la
  serie `target` en el día siguiente a la ventana.
  - `se_saben_antes[i] = True` ⇒ esa serie se **adelanta un día** (la conocemos
    para el día a predecir: calendario, festivos, promo…).
  - `se_saben_antes[i] = False` ⇒ serie endógena/desconocida; solo se usa su
    pasado. Los primeros `W_in` registros quedan con `NAN`.
- **`info_enventanado(...)`** — imprime de forma legible el contenido de las ventanas.
- **`NAN`** — `np.nan`.

### Arquitectura de referencia (notebook 03)

API funcional de Keras, **un `Input` por variable**:

```
endógena (Sales pasado)  ─┐
festivo / promo          ─┤
mes      ─► Embedding    ─┤
día sem. ─► Embedding    ─┼─► concatenate ─► LSTM/GRU ─► Dense(1) ─► venta predicha
StoreType─► Embedding    ─┤
...      ─► Embedding    ─┘
```

## 2. Adaptación a Rossmann (propuesta)

1. **Carga y unión**: `train.csv` + `store.csv` por `Store`; idem para `test.csv`.
2. **Limpieza**: tratar `Open == 0` (ventas 0 estructurales), nulos de
   `CompetitionDistance`/`Promo2*`.
3. **Ingeniería de variables**:
   - Endógena: `Sales` (log o estandarizada por tienda).
   - Categóricas → embeddings: `Store`, `StoreType`, `Assortment`,
     `DayOfWeek`, `StateHoliday`, `mes`, `Promo`, `SchoolHoliday`.
   - Numéricas escaladas: `CompetitionDistance`, (opcional) `Customers`.
   - Calendario derivado de `Date`: mes, día del mes, semana, año.
4. **Enventanado por tienda**: las ventanas no deben cruzar fronteras entre
   tiendas. El embedding de `Store` permite a una sola red aprender las 1.115
   series a la vez (modelo global) en lugar de una red por tienda.
5. **Split temporal**: train hasta una fecha de corte, validación con las
   últimas semanas; el test "oficial" es 2015-07-31.
6. **Horizonte**: por el hueco train(07-17)→test(07-31), evaluar la predicción
   **multi-step** (~2 semanas) además de la de 1 día.
7. **Baselines obligatorios** (para contextualizar el R²): modelo persistente
   (ventas de hoy = ventas de hace 7 días), media por tienda/día-de-semana.
8. **Comparativa de arquitecturas**: endógena sola → + exógenas → + embeddings;
   variar `W_in`, tamaño de LSTM/GRU, dimensión de embeddings, regularización.

## 3. Métrica

R² en test (`sklearn.metrics.r2_score`). Reportar R² por tienda y agregado, y
compararlo contra los baselines persistentes.

## 4. Dudas abiertas (para preguntar / confirmar)

1. **¿Qué son "las 10 tiendas"?** El enunciado evalúa el R² en test de **10
   tiendas**, pero los datos traen las **1.115**. Posibilidades:
   - hay una lista concreta de 10 tiendas que da el profesor,
   - elegimos nosotros 10 (¿representativas? ¿las 10 primeras?),
   - se evalúa el modelo global solo en esas 10.
   → **Hay que confirmarlo antes de fijar el split y la métrica.**
2. **¿Un modelo global (todas las tiendas con embedding de `Store`) o un modelo
   por tienda?** El global suele ganar y encaja con "entradas heterogéneas".
3. **¿Usamos `Customers` como exógena?** En test viene dado (atípico), pero en
   un escenario real no se conoce; conviene decidir si lo usamos o lo predecimos.

## 5. Checklist del entregable

- [ ] Portada con nombres del grupo.
- [ ] EDA y justificación de preprocesados.
- [ ] Baselines persistentes.
- [ ] Modelo endógeno → + exógenas → + embeddings (entradas heterogéneas).
- [ ] Tabla resumen de métricas (R²) por arquitectura.
- [ ] Visualización de embeddings (p. ej. tiendas / días).
- [ ] Reflexión final crítica.
