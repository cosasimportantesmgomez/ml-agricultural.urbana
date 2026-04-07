import argparse  # Aqui traemos argparse para leer parametros desde terminal sin enredos
from pathlib import Path  # Aqui usamos Path para manejar rutas de archivos mas ordenado

import matplotlib.pyplot as plt  # Esta libreria nos ayuda a dibujar la grafica final
import pandas as pd  # Pandas nos deja cargar y manipular la tabla del CSV facilito
import seaborn as sns  # Seaborn le da un estilo mas bonito a la grafica
from sklearn.ensemble import RandomForestRegressor  # Este es el modelo que aprende y luego predice
from sklearn.metrics import mean_squared_error, r2_score  # Estas funciones calculan que tan bien quedo el modelo
from sklearn.model_selection import train_test_split  # Esto parte los datos en entrenamiento y prueba


def entrenar_y_evaluar(ruta_dataset: Path, carpeta_salida: Path) -> None:  # Esta funcion hace todo el flujo de punta a punta
    datos = pd.read_csv(ruta_dataset)  # Aqui leemos el archivo CSV y lo dejamos en una tabla

    objetivo = datos["rendimiento"]  # Esta linea toma la columna que queremos predecir
    entradas = datos.drop("rendimiento", axis=1)  # Aqui nos quedamos con el resto de columnas para entrenar

    entradas = pd.get_dummies(entradas, drop_first=True)  # Aqui convertimos texto a numeros para que el modelo entienda

    entradas_entrenamiento, entradas_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(  # Aqui dividimos datos para entrenar y luego probar
        entradas,  # Estas son las columnas de entrada ya preparadas
        objetivo,  # Esta es la salida real que el modelo debe aprender
        test_size=0.2,  # Dejamos el 20 por ciento para prueba
        random_state=42,  # Fijamos semilla para que salga igual cada vez
    )  # Cerramos la division de datos

    modelo_bosque = RandomForestRegressor(  # Aqui creamos el modelo de bosque aleatorio
        n_estimators=300,  # Le decimos que use 300 arboles
        max_depth=10,  # Le ponemos profundidad maxima de 10 para controlar complejidad
        random_state=42,  # Semilla fija para repetir resultados
    )  # Cerramos la configuracion del modelo

    modelo_bosque.fit(entradas_entrenamiento, objetivo_entrenamiento)  # Aqui el modelo aprende con los datos de entrenamiento
    predicciones = modelo_bosque.predict(entradas_prueba)  # Aqui el modelo intenta adivinar con los datos de prueba

    error_medio_cuadratico = mean_squared_error(objetivo_prueba, predicciones)  # Esta metrica mide que tan lejos quedaron las predicciones
    puntaje_r2 = r2_score(objetivo_prueba, predicciones)  # Esta metrica dice que tan bien explica el modelo los datos

    print(f"MSE: {error_medio_cuadratico:.6f}")  # Mostramos el error en pantalla
    print(f"R2: {puntaje_r2:.6f}")  # Mostramos el puntaje R2 en pantalla

    carpeta_salida.mkdir(parents=True, exist_ok=True)  # Creamos carpeta de salida si aun no existe

    sns.set_theme(style="whitegrid")  # Ponemos un estilo limpio para la grafica
    plt.figure(figsize=(8, 5))  # Abrimos un lienzo de tamano comodo
    sns.scatterplot(x=objetivo_prueba, y=predicciones, alpha=0.75)  # Dibujamos puntos de real contra predicho

    valor_minimo = min(objetivo_prueba.min(), predicciones.min())  # Sacamos el valor mas pequeno para la linea guia
    valor_maximo = max(objetivo_prueba.max(), predicciones.max())  # Sacamos el valor mas grande para la linea guia
    plt.plot([valor_minimo, valor_maximo], [valor_minimo, valor_maximo], color="red", linestyle="--", linewidth=1)  # Dibujamos linea diagonal ideal

    plt.title("Real vs Predicho (Random Forest)")  # Titulo de la grafica
    plt.xlabel("Valor real")  # Nombre del eje horizontal
    plt.ylabel("Prediccion")  # Nombre del eje vertical
    plt.tight_layout()  # Ajustamos margenes para que se vea bien

    ruta_grafica = carpeta_salida / "real_vs_predicho.png"  # Armamos la ruta donde se guarda la imagen
    plt.savefig(ruta_grafica, dpi=150)  # Guardamos la grafica en archivo PNG
    plt.close()  # Cerramos la figura para liberar memoria

    print(f"Grafica guardada en: {ruta_grafica}")  # Avisamos en consola donde quedo la imagen


def leer_argumentos() -> argparse.Namespace:  # Esta funcion junta los parametros de ejecucion
    analizador = argparse.ArgumentParser(  # Creamos el lector de argumentos de terminal
        description="Entrena un RandomForestRegressor para predecir rendimiento agricola."  # Texto de ayuda de la app
    )  # Cerramos creacion del analizador
    analizador.add_argument(  # Aqui declaramos el parametro para la ruta del dataset
        "--data-path",  # Nombre del parametro en terminal
        default="data/dataset_agricultura_real_medellin.csv",  # Ruta por defecto del CSV
        help="Ruta del CSV con los datos (default: data/dataset_agricultura_real_medellin.csv)",  # Ayuda para el usuario
    )  # Cerramos definicion del parametro data-path
    analizador.add_argument(  # Aqui declaramos el parametro para carpeta de salida
        "--output-dir",  # Nombre del parametro en terminal
        default="outputs",  # Carpeta por defecto para salidas
        help="Directorio para guardar salidas (default: outputs)",  # Ayuda del parametro output-dir
    )  # Cerramos definicion del parametro output-dir
    return analizador.parse_args()  # Devolvemos los argumentos leidos


def principal() -> None:  # Esta funcion organiza el flujo principal del script
    argumentos = leer_argumentos()  # Leemos lo que se mando por terminal
    ruta_dataset = Path(argumentos.data_path)  # Convertimos la ruta del CSV a objeto Path
    carpeta_salida = Path(argumentos.output_dir)  # Convertimos la carpeta de salida a objeto Path
    entrenar_y_evaluar(ruta_dataset, carpeta_salida)  # Ejecutamos entrenamiento y evaluacion


if __name__ == "__main__":  # Esta condicion asegura que corra solo si ejecutas este archivo directo
    principal()  # Llamamos la funcion principal para arrancar todo
