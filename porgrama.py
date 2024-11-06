import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración del estilo global
st.set_page_config(layout="wide")
plt.style.use('dark_background')  # Cambiado de 'seaborn' a 'dark_background'

# Configuración del tema personalizado
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1, h2, h3 {
        color: #ff69b4 !important;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff69b4;
        border: none;
    }
    .stTextInput>div>div>input {
        color: #ff69b4;
    }
    div[data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    .css-145kmo2 {
        color: #ff69b4 !important;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("Métodos de Optimización")

# Menú lateral
ejercicio = st.sidebar.selectbox(
    "Selecciona un ejercicio",
    ["Ejercicio 1: Maximización de Datos Procesados",
     "Ejercicio 2: Optimización del Sistema Distribuido",
     "Ejercicio 3: Tiempo de Ejecución",
     "Ejercicio 4: Optimización del Uso de CPU",
     "Ejercicio 5: Minimización del Tiempo de Entrenamiento",
     "Ejercicio 6: Maximización de Archivos Transmitidos",
     "Ejercicio 7: Sistema de Colas",
     "Ejercicio 8: Deep Learning",
     "Ejercicio 9: Almacenamiento en la Nube",
     "Ejercicio 10: Sistema de Mensajería"]
)

def configurar_estilo_grafico():
    plt.rcParams['text.color'] = '#ffffff'
    plt.rcParams['axes.labelcolor'] = '#ff69b4'
    plt.rcParams['xtick.color'] = '#ff69b4'
    plt.rcParams['ytick.color'] = '#ff69b4'
    plt.rcParams['axes.titlecolor'] = '#ff69b4'

def ejercicio1():
    st.header("Maximización de Datos Procesados por Algoritmo")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        memoria_total = st.number_input("Capacidad total de memoria (MB)", value=1024, step=1)
        num_lotes = st.slider("Número de lotes", 1, 8, 8)
        x = st.number_input("Memoria requerida por lote (MB)", value=128, step=1)
    
    def calcular_datos_procesados(x, num_lotes):
        if num_lotes <= 5:
            return num_lotes * x
        else:
            return (5 * x) + ((num_lotes - 5) * 0.8 * x)
    
    with col2:
        if num_lotes * x <= memoria_total:
            datos_procesados = calcular_datos_procesados(x, num_lotes)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            configurar_estilo_grafico()
            
            lotes = np.arange(1, num_lotes + 1)
            datos = [calcular_datos_procesados(x, l) for l in lotes]
            
            ax.plot(lotes, datos, marker='o', color='#ff69b4', linestyle='-', linewidth=2, markersize=8)
            ax.set_xlabel("Número de lotes")
            ax.set_ylabel("Datos procesados (MB)")
            ax.set_title("Datos procesados vs Número de lotes")
            ax.grid(True, color='#666666')
            
            st.pyplot(fig)
            
            st.write(f"Datos procesados: {datos_procesados:.2f} MB")
        else:
            st.error("La memoria requerida excede la capacidad total.")

def ejercicio2():
    st.header("Optimización del Sistema Distribuido")
    
    num_nodos = 20
    capacidad_red = 400

    def funcion_objetivo(x):
        return 20 * x

    x_values = np.linspace(0, capacidad_red / num_nodos, 100)
    y_values = funcion_objetivo(x_values)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(f"""
        - Número de nodos: {num_nodos}
        - Capacidad de red: {capacidad_red} peticiones/segundo
        """)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        configurar_estilo_grafico()
        
        ax.plot(x_values, y_values, color='#ff69b4', linewidth=2)
        ax.set_xlabel("Peticiones por nodo")
        ax.set_ylabel("Total de peticiones procesadas")
        ax.set_title("Optimización del Sistema Distribuido")
        ax.grid(True, color='#666666')
        
        st.pyplot(fig)

def ejercicio3():
    st.header("Tiempo de Ejecución")

    def tiempo_ejecucion(x):
        return 5 * x + 2

    # Definir el tiempo límite
    tiempo_limite = 50

    # Calcular el número máximo de datos que se pueden procesar
    x_max = int((tiempo_limite - 2) / 5)
    x_min = 0  # Valor mínimo de datos

    # Mostrar el valor máximo de datos
    st.write(f"El número máximo de datos que el script puede procesar sin exceder {tiempo_limite} segundos es: **{x_max}**")

    # Generar los valores de x y la función de tiempo de ejecución para la gráfica
    x_values = np.arange(0, x_max + 1)
    y_values = tiempo_ejecucion(x_values)

    # Crear la gráfica con formato POM-QM
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, marker="o", linestyle="--", color="b")

    # Agregar detalles a la gráfica
    ax.set_title("Gráfico del Tiempo de Ejecución (T(x) = 5x + 2)", fontsize=16)
    ax.set_xlabel("Número de datos (x)", fontsize=12)
    ax.set_ylabel("Tiempo de ejecución (segundos)", fontsize=12)
    ax.grid(True)

    # Marcar el punto máximo y mínimo
    ax.scatter(x_max, tiempo_ejecucion(x_max), color="r", zorder=5, label=f'Máximo en x = {x_max}')
    ax.scatter(x_min, tiempo_ejecucion(x_min), color="g", zorder=5, label=f'Mínimo en x = {x_min}')
    ax.legend()

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)

def ejercicio4():
    def uso_cpu(x):
        return 2 * x ** 2 + 10 * x

    # Función para calcular el mínimo de la función en un rango dado
    def calcular_minimo(func, min_x, max_x):
        x_vals = np.linspace(min_x, max_x, 1000)
        y_vals = func(x_vals)
        min_index = np.argmin(y_vals)
        return x_vals[min_index], y_vals[min_index]

    # Función para calcular el máximo de la función en un rango dado
    def calcular_maximo(func, min_x, max_x):
        x_vals = np.linspace(min_x, max_x, 1000)
        y_vals = func(x_vals)
        max_index = np.argmax(y_vals)
        return x_vals[max_index], y_vals[max_index]

    # Configuración del encabezado y controles de la aplicación en Streamlit
    st.title('Optimización del Uso de la CPU')
    st.write("""
    Este programa permite visualizar la función de uso de CPU en función del número de peticiones por segundo,
    y muestra el valor mínimo y máximo en un rango específico.
    """)

    # Input del rango de peticiones por segundo
    min_x = st.number_input('Número mínimo de peticiones por segundo', value=1, step=1)
    max_x = st.number_input('Número máximo de peticiones por segundo', value=20, step=1)

    # Calcular los valores máximo y mínimo
    x_min, y_min = calcular_minimo(uso_cpu, min_x, max_x)
    x_max, y_max = calcular_maximo(uso_cpu, min_x, max_x)

    # Mostrar los resultados
    st.write(f"El mínimo uso de CPU ocurre cuando se procesan {x_min:.2f} peticiones por segundo, con un uso de {y_min:.2f}%.")
    st.write(f"El máximo uso de CPU ocurre cuando se procesan {x_max:.2f} peticiones por segundo, con un uso de {y_max:.2f}%.")

    # Generar la gráfica de la función
    x_vals = np.linspace(min_x, max_x, 1000)
    y_vals = uso_cpu(x_vals)

    # Crear la figura usando Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='Uso de CPU (2x^2 + 10x)', color='blue')
    plt.scatter([x_min], [y_min], color='red', label=f'Mínimo ({x_min:.2f}, {y_min:.2f})', zorder=5)
    plt.scatter([x_max], [y_max], color='green', label=f'Máximo ({x_max:.2f}, {y_max:.2f})', zorder=5)

    # Añadir títulos y etiquetas
    plt.title('Gráfica del uso de la CPU en función de las peticiones por segundo')
    plt.xlabel('Peticiones por segundo')
    plt.ylabel('Uso de CPU (%)')
    plt.legend()

    # Mostrar la gráfica en Streamlit
    st.pyplot(plt)

def ejercicio5():
    def T(x):
        return (1000 / x) + (0.1 * x)

    # Definir los valores de x en el rango permitido
    x_values = np.linspace(16, 128, 100)
    t_values = T(x_values)

    # Calcular el mínimo
    min_x = 100
    min_y = T(min_x)

    # Calcular el máximo en los extremos
    t_16 = T(16)
    t_128 = T(128)

    # Determinar el máximo y su posición
    if t_16 > t_128:
        max_x = 16
        max_y = t_16
    else:
        max_x = 128
        max_y = t_128

    # Configuración de la aplicación Streamlit
    st.title("Minimización del Tiempo de Entrenamiento")
    st.write("La función a minimizar es T(x) = (1000/x) + 0.1x")

    # Gráfico de la función
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, t_values, label='T(x)', color='blue')
    plt.scatter(min_x, min_y, color='red', zorder=5, label=f'Mínimo: T({min_x})={min_y}')
    plt.scatter(max_x, max_y, color='orange', zorder=5, label=f'Máximo: T({max_x})={max_y}')
    plt.axhline(min_y, color='red', linestyle='--', label='Mínimo')
    plt.axvline(min_x, color='red', linestyle='--')
    plt.axhline(max_y, color='orange', linestyle='--', label='Máximo')
    plt.axvline(max_x, color='orange', linestyle='--')
    plt.title("Gráfico de la Función de Tiempo de Entrenamiento")
    plt.xlabel("Batch Size (x)")
    plt.ylabel("Tiempo de Entrenamiento T(x)")
    plt.grid(True)
    plt.legend()
    plt.xlim(16, 128)
    plt.ylim(0, 70)
    plt.tight_layout()
    st.pyplot(plt)


def ejercicio6():
    total_bandwidth = 1000  # en Mbps
    max_files = 50           # máximo de archivos

    # Función para calcular el número máximo de archivos transmitidos
    def max_files_transmitted(x):
        n_values = []
        for n in range(1, max_files + 1):
            if n <= 30:
                if n * x <= total_bandwidth:
                    n_values.append(n)
            else:
                available_bandwidth = total_bandwidth * (1 - 0.05 * (n - 30))
                if n * x <= available_bandwidth:
                    n_values.append(n)
        return n_values

    # Definimos el rango de x (uso de ancho de banda por archivo)
    x_values = np.linspace(1, 50, 50)  # evitando 0 para no dividir por cero
    max_n_values = [max(max_files_transmitted(x)) for x in x_values]

    # Determinar el valor máximo
    max_n = max(max_n_values)
    max_x_index = np.argmax(max_n_values)
    optimal_x = x_values[max_x_index]

    # Determinar el valor mínimo (en este caso sería 0 si no hay archivos transmitidos)
    min_n = 0

    # Configuración de la aplicación Streamlit
    st.title("Maximización del Número de Archivos Transmitidos")
    st.write("El ancho de banda total es de 1000 Mbps y cada archivo utiliza x Mbps.")

    # Gráfico de la función
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, max_n_values, label='Número de Archivos Transmitidos', color='blue')
    plt.axvline(optimal_x, color='red', linestyle='--', label=f'Óptimo: x = {optimal_x:.2f} Mbps')
    plt.scatter(optimal_x, max_n, color='red', zorder=5, label=f'Máximo: {max_n} archivos')
    plt.scatter(1, min_n, color='green', zorder=5, label=f'Mínimo: {min_n} archivos (sin archivos)')
    plt.title("Gráfico del Número de Archivos Transmitidos vs. Uso de Ancho de Banda")
    plt.xlabel("Ancho de Banda por Archivo (x Mbps)")
    plt.ylabel("Número de Archivos Transmitidos")
    plt.grid(True)
    plt.legend()
    plt.xlim(1, 50)
    plt.ylim(0, max_n + 5)
    plt.tight_layout()
    st.pyplot(plt)

def ejercicio7():
    def T(x):
        return (100 / x) + (2 * x)

    x_values = np.linspace(5, 15, 400)
    y_values = T(x_values)

    x_min = 7.07
    y_min = T(x_min)
    x_max = 15
    y_max = T(x_max)

    st.title("Minimización del Tiempo de Respuesta del Sistema de Colas")

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label='T(x) = 100/x + 2x', color='blue')
    plt.scatter([5, x_min, x_max], [T(5), y_min, y_max], color='red')
    plt.annotate('T(5) = 30', (5, T(5)), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(f'Mínimo T({x_min:.2f}) ≈ {y_min:.2f}', (x_min, y_min), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(f'Máximo T({x_max}) = {y_max:.2f}', (x_max, y_max), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title("Gráfica del Tiempo de Respuesta T(x)")
    plt.xlabel("Trabajos por segundo (x)")
    plt.ylabel("Tiempo de respuesta (T)")
    plt.grid()
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.legend()
    st.pyplot(plt)

def ejercicio8():
    # Función de penalización para lotes mayores a 10
    def penalizacion(x):
        return 0.1 * (x - 10) if x > 10 else 0

    # Función de consumo de energía por lote
    def consumo_energia(x):
        return x + penalizacion(x)

    # Rango de valores de x (tamaño del lote)
    x_values = np.linspace(1, 30, 400)
    y_values = np.array([consumo_energia(x) for x in x_values])

    # Encontrar el mínimo y máximo consumo de energía
    x_min = 1  # Valor mínimo de x
    y_min = consumo_energia(x_min)
    x_max = 10  # Valor máximo de x que no incurre en penalización
    y_max = consumo_energia(x_max)

    st.title("Maximización del Tamaño del Lote en Deep Learning")

    # Gráfica de la función de consumo de energía
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label='Consumo de Energía por Lote', color='blue')
    plt.scatter([x_min, x_max], [y_min, y_max], color='red')  # Puntos mínimos y máximos relevantes
    plt.annotate(f'Mínimo en x = {x_min}, Consumo ≈ {y_min:.2f}', (x_min, y_min), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(f'Máximo en x = {x_max}, Consumo ≈ {y_max:.2f}', (x_max, y_max), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title("Consumo de Energía por Tamaño del Lote")
    plt.xlabel("Tamaño del Lote (x)")
    plt.ylabel("Consumo de Energía (Unidades)")
    plt.axhline(200, color='green', linestyle='--', label='Límite de Consumo (200 unidades)')
    plt.grid()
    plt.legend()
    st.pyplot(plt)


# Nueva función Ejercicio 9 (Maximización del Almacenamiento)
def ejercicio9():
    # Definir la función de costo
    def costo_almacenamiento(x):
        return 50 + 5 * x

    # Presupuesto total
    presupuesto = 500

    # Calcular la cantidad máxima de TB que se puede almacenar
    max_tb = (presupuesto - 50) / 5

    # Rango de valores de x (cantidad de TB)
    x_values = np.linspace(0, 100, 400)
    y_values = costo_almacenamiento(x_values)

    # Valores máximo y mínimo de la función
    costo_maximo = costo_almacenamiento(max_tb)  # Costo al límite de TB
    costo_minimo = costo_almacenamiento(0)       # Costo mínimo al usar 0 TB

    # Configuración de Streamlit
    st.title("Maximización del Almacenamiento en la Nube")

    # Gráfica de la función de costo
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label='Costo de Almacenamiento por TB', color='blue')
    plt.axhline(presupuesto, color='green', linestyle='--', label='Presupuesto ($500)')
    plt.axvline(max_tb, color='red', linestyle='--', label=f'Máx. Almacenamiento = {max_tb:.2f} TB')
    plt.scatter(max_tb, costo_maximo, color='orange', s=100, label=f'Máximo Costo = ${costo_maximo:.2f} (en {max_tb:.2f} TB)', zorder=5)
    plt.scatter(0, costo_minimo, color='purple', s=100, label=f'Mínimo Costo = ${costo_minimo:.2f} (en 0 TB)', zorder=5)
    plt.title("Costo de Almacenamiento vs. TB")
    plt.xlabel("Cantidad de Almacenamiento (TB)")
    plt.ylabel("Costo (dólares)")
    plt.xlim(0, 100)
    plt.ylim(0, presupuesto + 50)
    plt.grid()
    plt.legend()
    st.pyplot(plt)

def ejercicio10():
    # Definir la función de latencia
    def latencia(x):
        return 100 - 2 * x

    # Definir el límite de latencia
    latencia_minima = 20

    # Calcular la cantidad máxima de mensajes que se pueden enviar
    max_mensajes = (100 - latencia_minima) / 2  # Límite superior

    # Rango de valores de x (número de mensajes por segundo)
    x_values = np.linspace(0, 60, 400)  # Se extiende más allá del máximo para ver el efecto
    y_values = latencia(x_values)

    # Valores máximo y mínimo de la función
    latencia_maxima = latencia(0)  # Latencia en 0 mensajes
    latencia_minima_value = latencia(max_mensajes)  # Latencia en el límite de mensajes

    # Configuración de Streamlit
    st.title("Maximización del Número de Mensajes en un Sistema de Mensajería")

    # Gráfica de la función de latencia
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label='Latencia (ms)', color='blue')
    plt.axhline(latencia_minima, color='green', linestyle='--', label='Latencia Mínima (20 ms)')
    plt.axvline(max_mensajes, color='red', linestyle='--', label=f'Máx. Mensajes = {max_mensajes:.2f} mensajes/s')

    # Resaltar el valor mínimo y máximo
    plt.scatter(max_mensajes, latencia_minima_value, color='orange', s=100,
                label=f'Latencia Mínima = {latencia_minima_value:.2f} ms\n(en {max_mensajes:.2f} mensajes/s)', zorder=5)
    plt.scatter(0, latencia_maxima, color='purple', s=100,
                label=f'Latencia Máxima = {latencia_maxima:.2f} ms\n(en 0 mensajes/s)', zorder=5)

    # Añadir anotaciones para los puntos
    plt.annotate(f'Máxima: {latencia_maxima:.2f} ms', xy=(0, latencia_maxima), 
                 xytext=(5, latencia_maxima + 10), 
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Mínima: {latencia_minima_value:.2f} ms', xy=(max_mensajes, latencia_minima_value), 
                 xytext=(max_mensajes + 5, latencia_minima_value + 10), 
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title("Latencia vs. Número de Mensajes por Segundo")
    plt.xlabel("Número de Mensajes por Segundo")
    plt.ylabel("Latencia (ms)")
    plt.xlim(0, 60)
    plt.ylim(0, 100)
    plt.grid()
    plt.legend()
    st.pyplot(plt)  

# Definir las funciones para los ejercicios 4-10 de manera similar

# Diccionario de funciones
ejercicios = {
    "Ejercicio 1: Maximización de Datos Procesados": ejercicio1,
    "Ejercicio 2: Optimización del Sistema Distribuido": ejercicio2,
    "Ejercicio 3: Tiempo de Ejecución": ejercicio3,
    "Ejercicio 4: Optimización del Uso de CPU": ejercicio4,
    "Ejercicio 5: Minimización del Tiempo de Entrenamiento": ejercicio5,
    "Ejercicio 6: Maximización de Archivos Transmitidos": ejercicio6,
    "Ejercicio 7: Sistema de Colas": ejercicio7,
    "Ejercicio 8: Deep Learning": ejercicio8,
    "Ejercicio 9: Almacenamiento en la Nube": ejercicio9,
    "Ejercicio 10: Sistema de Mensajería": ejercicio10,
    # Agregar las demás funciones al diccionario
}

# Ejecutar el ejercicio seleccionado
if ejercicio in ejercicios:
    ejercicios[ejercicio]()
