from pathlib import Path  # Traemos Path para manejar rutas de archivos de forma simple

import matplotlib.pyplot as plt  # Importamos pyplot para dibujar graficas
import pandas as pd  # Importamos pandas para manejar datos en tablas
import seaborn as sns  # Seaborn mejora el estilo de las graficas
import streamlit as st  # Streamlit crea la interfaz interactiva
from sklearn.ensemble import RandomForestRegressor  # Modelo de regresion con bosque aleatorio
from sklearn.metrics import mean_squared_error, r2_score  # Metricas para evaluar el modelo
from sklearn.model_selection import train_test_split  # Divide datos para entrenar y probar


st.set_page_config(page_title="Prediccion Agricola", page_icon="🌱", layout="wide")  # Configuramos pagina completa


def aplicar_estilos() -> None:  # Funcion para dar estilo moderno a la app
    st.markdown(  # Inyectamos CSS personalizado
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');

            :root {
                --fondo: #f4f8f3;
                --texto: #1e2c24;
                --verde: #1f6f52;
                --verde-claro: #2f8f6a;
                --ambar: #efb93b;
                --tarjeta: #ffffff;
            }

            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(1100px 500px at 6% -10%, rgba(31,111,82,0.16), transparent 55%),
                    radial-gradient(800px 400px at 95% 0%, rgba(239,185,59,0.20), transparent 50%),
                    var(--fondo);
                color: var(--texto);
            }

            .hero {
                background: linear-gradient(135deg, var(--verde), var(--verde-claro));
                color: #ffffff;
                border-radius: 18px;
                padding: 1.35rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 14px 28px rgba(31, 111, 82, 0.24);
            }

            .hero h1 {
                margin: 0;
                font-size: 1.8rem;
                font-weight: 800;
            }

            .hero p {
                margin: 0.4rem 0 0;
                font-size: 1rem;
                opacity: 0.92;
            }

            .kpi {
                background: var(--tarjeta);
                border-radius: 14px;
                border: 1px solid rgba(31,111,82,0.12);
                box-shadow: 0 8px 20px rgba(0,0,0,0.06);
                padding: 0.9rem 1rem;
                margin-bottom: 0.85rem;
            }

            .kpi .etiqueta {
                color: #4a5a52;
                font-weight: 700;
                letter-spacing: 0.03em;
                font-size: 0.82rem;
                text-transform: uppercase;
            }

            .kpi .valor {
                color: var(--verde);
                font-size: 1.45rem;
                font-weight: 800;
                margin-top: 0.2rem;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
            }

            .stTabs [data-baseweb="tab"] {
                background: #e7efe8;
                border-radius: 12px;
                padding: 0.45rem 0.95rem;
                font-weight: 700;
            }

            .stTabs [aria-selected="true"] {
                background: var(--verde) !important;
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )  # Cerramos CSS


@st.cache_data  # Guardamos los datos en cache para no recargar cada vez
def cargar_datos(ruta_dataset: str) -> pd.DataFrame:  # Funcion que lee el CSV
    return pd.read_csv(ruta_dataset)  # Devolvemos datos del archivo


@st.cache_resource  # Guardamos el modelo en cache para no reentrenar cada accion
def entrenar_flujo(datos: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, list[str]]:  # Funcion que entrena y devuelve partes utiles
    objetivo = datos["rendimiento"]  # Columna objetivo
    entradas = datos.drop("rendimiento", axis=1)  # Variables de entrada crudas
    entradas_codificadas = pd.get_dummies(entradas, drop_first=True)  # Convertimos texto a numeros igual que en Colab

    entradas_entrenamiento, entradas_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(
        entradas_codificadas,
        objetivo,
        test_size=0.2,
        random_state=42,
    )  # Dividimos datos

    modelo = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
    )  # Creamos el modelo igual al enfoque original

    modelo.fit(entradas_entrenamiento, objetivo_entrenamiento)  # Entrenamos modelo
    columnas_modelo = entradas_codificadas.columns.tolist()  # Guardamos columnas usadas al entrenar
    return modelo, entradas_prueba, objetivo_prueba, entradas, datos, columnas_modelo  # Devolvemos todo lo necesario para interfaz


def codificar_y_alinear(entradas: pd.DataFrame, columnas_modelo: list[str]) -> pd.DataFrame:  # Funcion para que cualquier entrada tenga las mismas columnas del entrenamiento
    entradas_codificadas = pd.get_dummies(entradas, drop_first=True)  # Aplicamos get_dummies igual que en entrenamiento
    entradas_alineadas = entradas_codificadas.reindex(columns=columnas_modelo, fill_value=0)  # Alineamos columnas faltantes/sobrantes
    return entradas_alineadas  # Devolvemos entradas listas para predecir


def pintar_kpi(etiqueta: str, valor: str) -> None:  # Funcion para mostrar tarjetas de metrica
    st.markdown(
        f"""
        <div class="kpi">
            <div class="etiqueta">{etiqueta}</div>
            <div class="valor">{valor}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )  # Cerramos tarjeta


def construir_formulario_manual(entradas: pd.DataFrame) -> dict:  # Funcion que construye formulario dinamico
    entrada_manual = {}  # Diccionario para guardar entradas del usuario

    for columna in entradas.columns:  # Recorremos cada columna
        if pd.api.types.is_numeric_dtype(entradas[columna]):  # Si es numerica
            valor_minimo = float(entradas[columna].min())
            valor_maximo = float(entradas[columna].max())
            valor_promedio = float(entradas[columna].mean())
            paso = (valor_maximo - valor_minimo) / 100 if valor_maximo != valor_minimo else 1.0
            entrada_manual[columna] = st.number_input(
                columna,
                min_value=valor_minimo,
                max_value=valor_maximo,
                value=valor_promedio,
                step=paso,
            )
        else:  # Si es categorica
            opciones = sorted(entradas[columna].dropna().astype(str).unique().tolist())
            if opciones:
                entrada_manual[columna] = st.selectbox(columna, options=opciones)
            else:
                entrada_manual[columna] = st.text_input(columna, value="")

    return entrada_manual  # Devolvemos entradas


def principal() -> None:  # Funcion principal
    aplicar_estilos()  # Aplicamos estilo visual

    st.markdown(
        """
        <div class="hero">
            <h1>Prediccion de Rendimiento Agricola</h1>
            <p>Interfaz moderna para explorar cultivos, analizar resultados y crear predicciones al instante.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )  # Mostramos bloque superior

    st.sidebar.header("Configuracion")  # Titulo lateral
    ruta_por_defecto = Path("data/dataset_agricultura_real_medellin.csv")  # Ruta por defecto
    ruta_datos = st.sidebar.text_input("Ruta del dataset CSV", str(ruta_por_defecto))  # Campo editable de ruta

    modelo, entradas_prueba, objetivo_prueba, entradas_completas, datos, columnas_modelo = entrenar_flujo(cargar_datos(ruta_datos))  # Cargamos y entrenamos

    predicciones = modelo.predict(entradas_prueba)  # Predecimos sobre prueba
    error_medio_cuadratico = mean_squared_error(objetivo_prueba, predicciones)  # MSE
    puntaje_r2 = r2_score(objetivo_prueba, predicciones)  # R2

    c1, c2, c3, c4 = st.columns(4)  # Cuatro columnas para tarjetas
    with c1:
        pintar_kpi("MSE", f"{error_medio_cuadratico:.4f}")
    with c2:
        pintar_kpi("R2", f"{puntaje_r2:.4f}")
    with c3:
        pintar_kpi("Registros", f"{len(datos)}")
    with c4:
        pintar_kpi("Variables", f"{len(entradas_completas.columns)}")

    izquierda, derecha = st.columns([1.8, 1.2])  # Distribucion para dos graficas

    with izquierda:
        st.subheader("Real vs Predicho")
        fig_real_pred, ax_real_pred = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=objetivo_prueba, y=predicciones, alpha=0.75, ax=ax_real_pred, color="#1f6f52")
        minimo = min(objetivo_prueba.min(), predicciones.min())
        maximo = max(objetivo_prueba.max(), predicciones.max())
        ax_real_pred.plot([minimo, maximo], [minimo, maximo], color="#d1495b", linestyle="--", linewidth=1.2)
        ax_real_pred.set_xlabel("Valor real")
        ax_real_pred.set_ylabel("Prediccion")
        ax_real_pred.set_title("Comparacion del modelo")
        st.pyplot(fig_real_pred)

    with derecha:
        st.subheader("Distribucion del Error")
        errores = objetivo_prueba - predicciones
        fig_error, ax_error = plt.subplots(figsize=(6, 5))
        sns.histplot(errores, bins=16, kde=True, ax=ax_error, color="#efb93b")
        ax_error.set_xlabel("Error (real - predicho)")
        ax_error.set_ylabel("Frecuencia")
        ax_error.set_title("Comportamiento de errores")
        st.pyplot(fig_error)

    pestana_1, pestana_2 = st.tabs(["Ver cultivo", "Nueva prediccion"])  # Creamos pestañas

    with pestana_1:
        st.subheader("Analisis por cultivo")
        if "cultivo" in datos.columns:
            cultivo_elegido = st.selectbox("Selecciona un cultivo", sorted(datos["cultivo"].astype(str).unique()))
            datos_cultivo = datos[datos["cultivo"].astype(str) == cultivo_elegido].copy()

            st.write(f"Registros encontrados: {len(datos_cultivo)}")
            st.dataframe(datos_cultivo.head(20), use_container_width=True, height=320)

            entradas_cultivo = datos_cultivo.drop("rendimiento", axis=1)
            entradas_cultivo_alineadas = codificar_y_alinear(entradas_cultivo, columnas_modelo)
            predicciones_cultivo = modelo.predict(entradas_cultivo_alineadas)

            promedio_real = datos_cultivo["rendimiento"].mean()
            promedio_predicho = float(predicciones_cultivo.mean())

            cx1, cx2 = st.columns(2)
            cx1.metric("Promedio real", f"{promedio_real:.4f}")
            cx2.metric("Promedio predicho", f"{promedio_predicho:.4f}")

            fig_cultivo, ax_cultivo = plt.subplots(figsize=(7, 3.8))
            serie_cultivo = datos_cultivo.reset_index(drop=True)
            sns.lineplot(data=serie_cultivo, x=serie_cultivo.index, y="rendimiento", ax=ax_cultivo, color="#2f8f6a")
            ax_cultivo.set_xlabel("Indice de registro")
            ax_cultivo.set_ylabel("Rendimiento")
            ax_cultivo.set_title("Tendencia de rendimiento del cultivo")
            st.pyplot(fig_cultivo)
        else:
            st.info("No existe la columna 'cultivo' en el dataset.")

    with pestana_2:
        st.subheader("Crear prediccion manual")
        entrada_usuario = construir_formulario_manual(entradas_completas)

        if st.button("Predecir rendimiento"):
            tabla_entrada = pd.DataFrame([entrada_usuario])
            tabla_entrada_alineada = codificar_y_alinear(tabla_entrada, columnas_modelo)
            prediccion_final = float(modelo.predict(tabla_entrada_alineada)[0])
            st.success(f"Rendimiento estimado: {prediccion_final:.4f}")
            st.balloons()
            st.dataframe(tabla_entrada, use_container_width=True)


if __name__ == "__main__":  # Punto de entrada
    principal()  # Ejecutamos app
