from pathlib import Path  # Traemos Path para manejar rutas de archivos sin necesidad de complicarse

import matplotlib.pyplot as plt  # Importamos pyplot para hacer graficas bonitas
import pandas as pd  # Importamos pandas para leer y manipular datos tipo tabla
import seaborn as sns  # Seaborn nos ayuda a hacer graficas mas atractivas visualmente
import streamlit as st  # Streamlit es lo que nos crea la interfaz web interactiva
from sklearn.compose import ColumnTransformer  # Esta herramienta nos deja preparar columnas diferentes de formas distintas
from sklearn.ensemble import RandomForestRegressor  # El modelo que aprende y predice rendimientos
from sklearn.metrics import mean_squared_error, r2_score  # Funciones para saber que tan bien predice el modelo
from sklearn.model_selection import train_test_split  # Divide datos entre entrenamieto y prueba
from sklearn.pipeline import Pipeline  # Pipeline une todos los pasos de preparacion y modelo en uno
from sklearn.preprocessing import OneHotEncoder  # Convierte texto a numeros para el modelo


st.set_page_config(page_title="Prediccion Agricola", page_icon="🌱", layout="wide")  # Configuramos la pagina: titulo, icono y que use todo el ancho


@st.cache_data  # Este decorador hace que los datos cargados se queden en memoria para no releer el CSV cada vez
def cargar_datos(ruta_dataset: str) -> pd.DataFrame:  # Funcion que lee el CSV y nos lo devuelve
    return pd.read_csv(ruta_dataset)  # Abrimos el archivo CSV y lo metemos en un dataframe


@st.cache_resource  # Este decorador hace que el modelo entrenado se quede en memoria sin reentrenar
def entrenar_flujo(datos: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series]:  # Funcion que prepara todo y entrena el modelo
    objetivo = datos["rendimiento"]  # Sacamos la columna que queremos predecir
    entradas = datos.drop("rendimiento", axis=1)  # Nos quedamos con el resto de columnas como entrada

    columnas_numericas = entradas.select_dtypes(include=["number"]).columns.tolist()  # Identificamos cuales columnas son numeros
    columnas_categoricas = [col for col in entradas.columns if col not in columnas_numericas]  # Identificamos cuales son texto

    procesador = ColumnTransformer(  # Este procesador va a preparar las columnas antes de entrenar
        transformers=[  # Aqui metemos las transformaciones
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),  # Convertimos texto a numeros
        ],  # Cerramos lista de transformaciones
        remainder="passthrough",  # Dejamos las columnas numericas como estan
    )  # Cerramos procesador

    modelo = Pipeline(  # Armamos un flujo que une procesamiento mas modelo
        steps=[  # Pasos del flujo
            ("procesador", procesador),  # Primer paso: procesar datos
            (  # Segundo paso: entrenar
                "regresor",  # Nombre del paso
                RandomForestRegressor(  # Creamos el modelo bosque aleatorio
                    n_estimators=300,  # 300 arboles de decision
                    max_depth=10,  # Profundidad maxima de 10
                    random_state=42,  # Semilla fija
                ),  # Cerramos configuracion del modelo
            ),  # Cerramos segundo paso
        ]  # Cerramos pasos
    )  # Cerramos flujo

    entradas_entrenamiento, entradas_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(  # Dividimos los datos
        entradas,  # Datos de entrada
        objetivo,  # Datos objetivo
        test_size=0.2,  # 20 por ciento para prueba
        random_state=42,  # Semilla fija
    )  # Cerramos division

    modelo.fit(entradas_entrenamiento, objetivo_entrenamiento)  # Aqui el modelo aprende con entrenamiento
    return modelo, entradas_prueba, objetivo_prueba, entradas  # Devolvemos el modelo y datos de prueba


def construir_formulario_manual(entradas: pd.DataFrame) -> dict:  # Funcion que arma el formulario dinamico en la interfaz
    entrada_manual = {}  # Diccionario para guardar lo que el usuario ingresa

    for columna in entradas.columns:  # Iteramos sobre cada columna del dataset
        if pd.api.types.is_numeric_dtype(entradas[columna]):  # Si la columna es numero
            valor_minimo = float(entradas[columna].min())  # Sacamos el minimo de la columna
            valor_maximo = float(entradas[columna].max())  # Sacamos el maximo de la columna
            valor_promedio = float(entradas[columna].mean())  # Sacamos el promedio
            paso_valor = (valor_maximo - valor_minimo) / 100 if valor_maximo != valor_minimo else 1.0  # Calculamos paso para el deslizador
            entrada_manual[columna] = st.number_input(  # Creamos un campo de numero en la interfaz
                columna,  # Le ponemos el nombre de la columna
                min_value=valor_minimo,  # Minimo valor permitido
                max_value=valor_maximo,  # Maximo valor permitido
                value=valor_promedio,  # Valor por defecto
                step=paso_valor,  # Paso para subir y bajar
            )  # Cerramos campo numero
        else:  # Si no es numero, es texto
            opciones = sorted(entradas[columna].dropna().astype(str).unique().tolist())  # Sacamos valores unicos en orden
            if opciones:  # Si hay opciones
                entrada_manual[columna] = st.selectbox(columna, options=opciones)  # Creamos dropdown para elegir
            else:  # Si no hay opciones
                entrada_manual[columna] = st.text_input(columna, value="")  # Creamos campo de texto libre

    return entrada_manual  # Devolvemos lo que ingreso el usuario


def principal() -> None:  # Funcion principal que organiza toda la interfaz
    st.title("Prediccion de Rendimiento Agricola")  # Titulo principal de la pagina
    st.write("Interfaz interactiva para explorar cultivos y generar nuevas predicciones.")  # Descripcion

    ruta_por_defecto = Path("data/dataset_agricultura_real_medellin.csv")  # Ruta esperada del CSV
    ruta_datos = st.sidebar.text_input("Ruta del dataset CSV", str(ruta_por_defecto))  # Campo para cambiar ruta en barra lateral

    datos = cargar_datos(ruta_datos)  # Cargamos los datos
    modelo, entradas_prueba, objetivo_prueba, entradas_completas = entrenar_flujo(datos)  # Entrenamos el modelo

    predicciones = modelo.predict(entradas_prueba)  # Hacemos predicciones con datos de prueba
    error_medio_cuadratico = mean_squared_error(objetivo_prueba, predicciones)  # Calculamos metrica de error
    puntaje_r2 = r2_score(objetivo_prueba, predicciones)  # Calculamos metrica de desempenio

    columna_1, columna_2 = st.columns(2)  # Dividimos pantalla en 2 columnas
    columna_1.metric("MSE", f"{error_medio_cuadratico:.4f}")  # Mostramos MSE en primera columna
    columna_2.metric("R2", f"{puntaje_r2:.4f}")  # Mostramos R2 en segunda columna

    st.subheader("Real vs Predicho")  # Subtitulo para seccion de grafica
    figura, grafico = plt.subplots(figsize=(8, 5))  # Creamos figura para la grafica
    sns.scatterplot(x=objetivo_prueba, y=predicciones, alpha=0.75, ax=grafico)  # Dibujamos puntos
    valor_minimo_grafico = min(objetivo_prueba.min(), predicciones.min())  # Minimo para linea de referencia
    valor_maximo_grafico = max(objetivo_prueba.max(), predicciones.max())  # Maximo para linea de referencia
    grafico.plot([valor_minimo_grafico, valor_maximo_grafico], [valor_minimo_grafico, valor_maximo_grafico], color="red", linestyle="--", linewidth=1)  # Dibujamos linea ideal
    grafico.set_xlabel("Valor real")  # Nombre eje X
    grafico.set_ylabel("Prediccion")  # Nombre eje Y
    grafico.set_title("Comparacion de valores")  # Titulo grafica
    st.pyplot(figura)  # Mostramos grafica en streamlit

    pestana_1, pestana_2 = st.tabs(["Ver cultivo", "Nueva prediccion"])  # Creamos 2 pestanas en la interfaz

    with pestana_1:  # Dentro primera pestana
        st.subheader("Analisis por cultivo")  # Titulo de la seccion
        if "cultivo" in datos.columns:  # Si existe columna cultivo
            cultivo_elegido = st.selectbox("Selecciona un cultivo", sorted(datos["cultivo"].astype(str).unique()))  # Dropdown para elegir cultivo
            subset = datos[datos["cultivo"].astype(str) == cultivo_elegido].copy()  # Filtramos datos por cultivo

            st.write(f"Registros encontrados: {len(subset)}")  # Mostramos cuantos registros hay
            st.dataframe(subset.head(20), use_container_width=True)  # Mostramos tabla con primeros 20 registros

            entradas_subset = subset.drop("rendimiento", axis=1)  # Sacamos columna objetivo del subset
            predicciones_subset = modelo.predict(entradas_subset)  # Predecimos con ese subset

            promedio_real = subset["rendimiento"].mean()  # Calculamos promedio real
            promedio_predicho = float(predicciones_subset.mean())  # Calculamos promedio predicho

            col_1, col_2 = st.columns(2)  # Dividimos en 2 columnas para metricas
            col_1.metric("Promedio real", f"{promedio_real:.4f}")  # Mostramos promedio real
            col_2.metric("Promedio predicho", f"{promedio_predicho:.4f}")  # Mostramos promedio predicho
        else:  # Si no existe columna cultivo
            st.info("No existe la columna 'cultivo' en el dataset.")  # Avisamos que falta columna

    with pestana_2:  # Dentro segunda pestana
        st.subheader("Crear prediccion manual")  # Titulo de la seccion
        entrada_usuario = construir_formulario_manual(entradas_completas)  # Armamos formulario dinamico

        if st.button("Predecir rendimiento"):  # Si usuario aprieta boton
            tabla_entrada = pd.DataFrame([entrada_usuario])  # Armamos tabla con datos del usuario
            prediccion_final = float(modelo.predict(tabla_entrada)[0])  # Predecimos rendimiento
            st.success(f"Rendimiento estimado: {prediccion_final:.4f}")  # Mostramos resultado en pantalla


if __name__ == "__main__":  # Si ejecutas este archivo directamente
    principal()  # Llama la funcion principal
