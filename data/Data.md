# Diccionario de datos — Rossmann Store Sales

Datos de la competición de Kaggle [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales).
La cadena Rossmann opera ~1.115 droguerías; el objetivo es predecir las
**ventas diarias** (`Sales`) por tienda.

## Ficheros

### `train.zip` → `train.csv`  (1.001.599 filas)

Histórico de ventas diarias por tienda. Rango temporal **2013-01-01 →
2015-07-17**, 1.115 tiendas. *(Distribuido comprimido por tamaño; `unzip train.zip`.)*

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `Store` | int | Identificador de la tienda (1–1115). |
| `DayOfWeek` | int | Día de la semana (1 = lunes … 7 = domingo). |
| `Date` | fecha | Fecha del registro. |
| `Sales` | int | **Variable objetivo**: ventas del día (0 … 41.551; media ≈ 5.771). |
| `Customers` | int | Número de clientes ese día. |
| `Open` | int | 1 = tienda abierta, 0 = cerrada. Si `Open == 0` ⇒ `Sales == 0` (170.627 filas cerradas). |
| `Promo` | int | 1 = la tienda tenía promoción ese día. |
| `StateHoliday` | cat | Festivo oficial: `0` = ninguno, `a` = festivo público, `b` = Semana Santa, `c` = Navidad. |
| `SchoolHoliday` | int | 1 = afectada por cierre de colegios. |
| `Id` | int | Identificador de fila. |

### `store.csv`  (1.115 filas)

Metadatos **estáticos** de cada tienda (se unen a train/test por `Store`).

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `Store` | int | Identificador de la tienda. |
| `StoreType` | cat | Tipo de tienda: `a` (602), `b` (17), `c` (148), `d` (348). |
| `Assortment` | cat | Nivel de surtido: `a` básico (593), `b` extra (9), `c` extendido (513). |
| `CompetitionDistance` | float | Distancia (m) a la competencia más cercana. 3 valores nulos. |
| `CompetitionOpenSinceMonth` / `...Year` | float | Mes/año de apertura de la competencia (con nulos). |
| `Promo2` | int | 1 = la tienda participa en la promoción continua "Promo2" (571 sí / 544 no). |
| `Promo2SinceWeek` / `...Year` | float | Semana/año de alta en Promo2 (nulos si `Promo2 == 0`). |
| `PromoInterval` | cat | Meses en que se renueva Promo2 (`Jan,Apr,Jul,Oct`, etc.; nulo si no participa). |

### `test.csv`  (1.115 filas)

Una fila **por tienda** para el día **2015-07-31**. No incluye `Sales` (es lo
que hay que predecir). Mismas columnas que train salvo `Sales`. Incluye
`Customers` (atención: en la competición original de Kaggle el test NO trae
clientes; aquí sí). 1.113 tiendas abiertas, 2 cerradas.

### `submission.csv`  (1.115 filas)

Formato de entrega de ejemplo: columnas `Id` y `Sales` (predicción).

## Notas de preprocesado relevantes

- **Días cerrados** (`Open == 0`) tienen `Sales == 0`: conviene tratarlos aparte
  (filtrarlos del entrenamiento o modelarlos explícitamente) para no contaminar
  la serie con ceros estructurales.
- **Hueco temporal**: train acaba el 2015-07-17 y test es 2015-07-31 ⇒ el
  modelo debe predecir ~2 semanas por delante (multi-step), no a 1 día.
- **Categóricas para embeddings**: `Store`, `StoreType`, `Assortment`,
  `DayOfWeek`, `StateHoliday`, mes, etc. — candidatas naturales a `Embedding`.
- **Numéricas a escalar**: `Sales` (endógena), `Customers`, `CompetitionDistance`.
- **Variables conocidas de antemano** (calendario, promo, festivos): se pueden
  "adelantar" en el enventanado (`se_saben_antes=True`) porque para el día a
  predecir ya las conocemos.
