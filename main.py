import numpy as np
import json
import os
from tkinter import filedialog, Tk
from Neural_network.metrics import show_error
from Neural_network.neural_network import Neural_network
from Neural_network.metrics import accuracy

current_network = None
data, labels, input_shape = None, None, None
images_2d = None  


def select_file(title="Seleccionar archivo", filetypes=(("Todos", "*.*"),)):
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def configuration():
    print("Seleccione el dataset")
    path = select_file("Seleccionar dataset", filetypes=[("NumPy", "*.npz")])

    if not path:
        print("No se selecciono ningun archivo.")
        return None, None, None, None

    if not os.path.exists(path):
        print(f"Error: {path} no encontrado.")
        return None, None, None, None

    try:
        d = np.load(path)
        images = d["images"]
        labels = d["labels"]

        images_norm = images / 255.0                          
        cantidad_datos = images_norm.shape[0]
        images_flat = images_norm.reshape(cantidad_datos, -1) 

        images_2d = images_norm if images_norm.ndim == 3 else images_norm

        print(f"Dataset cargado: {cantidad_datos} muestras, input={images_flat.shape[1]}")
        return images_flat, labels, images_flat.shape[1], images_2d

    except Exception as e:
        print(f"Error al procesar dataset: {e}")
        return None, None, None, None


def load_neural_network(file: str):
    try:
        with open(file, "r") as f:
            model_data = json.load(f)
        nn = Neural_network.from_dict(model_data)
        print("Red neuronal cargada exitosamente.")
        return nn
    except Exception as e:
        print(f"Error al cargar: {e}")
        return None


def save_neural_network(nn, file: str):
    try:
        with open(file, "w") as f:
            json.dump(nn.to_dict(), f, indent=4)
        print(f"Red neuronal guardada en {file}.")
    except Exception as e:
        print(f"Error al guardar: {e}")


def get_int(peticion, min_val=1):
    while True:
        try:
            val = int(input(peticion))
            if val >= min_val:
                return val
            print(f"Debe ser al menos {min_val}.")
        except ValueError:
            print("Ingresa un numero entero valido.")


def create_network():
    global input_shape, labels
    if input_shape is None:
        print("Primero carga un dataset (opcion 1).")
        return None

    num_clases = len(np.unique(labels))

    num_capas = get_int("\nIngrese la cantidad de capas (minimo 2): ", min_val=2)
    neurons_list = []
    act_func_list = []

    for i in range(num_capas):
        print(f"\n--- Capa {i + 1} ---")

        if i == num_capas - 1:
            print(f"Capa de salida: neuronas fijas a {num_clases} (clases detectadas)")
            neurons_list.append(num_clases)
            act_func_list.append("softmax")
        else:
            n = get_int("Cantidad de neuronas: ", min_val=1)
            neurons_list.append(n)
            print("Activacion: 1. ReLU")
            get_int("Opcion: ", min_val=1)   
            act_func_list.append("relu")

    nn = Neural_network(input_shape, neurons_list, act_func_list, labels)
    print(f"\nRed creada: {[input_shape] + neurons_list}")
    return nn


def main():
    global current_network, data, labels, input_shape, images_2d

    data, labels, input_shape, images_2d = configuration()

    while True:
        print("\n========== Meni Principal ==========")
        print(f"Estado: Dataset {'OK' if data is not None else 'Vacio'} | "  f"Red {'Cargada' if current_network else 'None'}")
        print("1. Cargar/Recargar dataset")
        print("2. Crear red neuronal")
        print("3. Cargar red neuronal (JSON)")
        print("4. Guardar red neuronal (JSON)")
        print("5. Entrenar")
        print("6. Predecir")
        print("7. Ver errores de prediccion")
        print("8. Salir")

        opcion = get_int("Seleccione: ")

        if opcion == 1:
            data, labels, input_shape, images_2d = configuration()

        elif opcion == 2:
            current_network = create_network()

        elif opcion == 3:
            print("Seleccione el modelo a cargar")
            path = select_file("Cargar modelo", filetypes=[("JSON", "*.json")])
            if path:
                current_network = load_neural_network(path)


        elif opcion == 4:
            if not current_network:
                print("No hay red cargada.")
            else:
                path = input("Nombre del archivo a guardar: ").strip()
                if path:
                    save_neural_network(current_network, path)


        elif opcion == 5:
            print("No se entrenar la red")


        elif opcion == 6:
            if not current_network or data is None:
                print("No hay red o datos.")
            else:
                probabilities = current_network.use(data)
                predictions = np.argmax(probabilities, axis=1)

                print("\n--- Resultados de la Prediccion ---")
                print(f"Predicciones:     {predictions[:20]}")
                print(f"Etiquetas reales: {labels[:20]}")
                accy = accuracy(probabilities, labels)
                print(f"Precision actual: {accy * 100:.2f}%")

        elif opcion == 7:
            if not current_network or data is None:
                print("No hay red o datos.")
            elif images_2d is None:
                print("No hay imagenes disponibles para visualizar.")
            else:
                probabilities = current_network.use(data)
                accy = accuracy(probabilities, labels)
                print(f"Precision actual: {accy * 100:.2f}%")
                num = get_int("¿Cuantos errores mostrar? (max recomendado 10): ")
                show_error(images_2d, labels, probabilities, num_errors=num)

        elif opcion == 8:
            print("Saliendo...")
            break

        else:
            print("Opcion no valida.")


if __name__ == "__main__":
    main()