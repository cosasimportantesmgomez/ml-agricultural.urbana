# Prediccion de Rendimiento Agricola - Machine Learning

Modelo de Machine Learning que predice rendimiento agrícola usando Random Forest. Interfaz interactiva con Streamlit para explorar datos y generar predicciones.

## Estructura del Proyecto

```
├── streamlit_app.py          # Aplicacion principal (lo unico que necesitas ejecutar)
├── train_model.py            # Script de entrenamiento (opcional)
├── requirements.txt          # Librerias necesarias
├── data/                     # Carpeta con los datos
│   └── dataset_agricultura_real_medellin.csv
├── outputs/                  # Resultados y graficas
└── README.md                 # Este archivo
```

## Requisitos

- Python 3.10 o superior (recomendado 3.12)

## Instalacion Paso a Paso

### 1. Clonar o descargar el proyecto

```powershell
git clone https://github.com/TU_USUARIO/ml-prediccion-agricola.git
cd ml-prediccion-agricola
```

### 2. Crear entorno virtual

```powershell
python -m venv .venv
```

### 3. Activar entorno virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell lo bloquea, ejecuta una sola vez:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4. Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

### 5. Agrega el dataset

Coloca tu archivo CSV en:
```
data/dataset_agricultura_real_medellin.csv
```

⚠️ El CSV debe tener la columna `rendimiento` como variable objetivo.

## Ejecutar la Aplicacion

```powershell
python -m streamlit run streamlit_app.py
```

Se abrira automaticamente en tu navegador (http://localhost:8501).

## Que puedes hacer en la app

✅ **Ver metricas del modelo**
- MSE (Error Cuadrático Medio)
- R² (Desempenio del modelo)
- Grafica Real vs Predicho

✅ **Explorar por cultivo**
- Filtra datos por tipo de cultivo
- Ve promedios reales vs predichos
- Visualiza registros del dataset

✅ **Hacer predicciones**
- Ingresa valores manualmente
- Obtén prediccion de rendimiento inmediato
- Genera predicciones sin necesidad de reentrenar

## Notas Importantes

- El modelo se entrena automaticamente al ejecutar la app por primera vez.
- Los datos se cachean en memoria para que sea rapido.
- No necesitas ejecutar `train_model.py` (solo si quieres guardar graficas en archivo).
- La interfaz es interactiva y responde en tiempo real.

## Ejecutar con otra ruta

```powershell
.\.venv\Scripts\python.exe train_model.py --data-path "data/tu_archivo.csv" --output-dir "outputs"
```

## Salidas

- Muestra en consola:
  - `MSE`
  - `R2`
- Genera una grafica en:
  - `outputs/real_vs_predicho.png`
