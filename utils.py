import json
import numpy as np
from tkinter import filedialog, Tk


def get_int(peticion: str, min_val: int = 1) -> int:
    while True:
        try:
            val = int(input(peticion))
            if val >= min_val:
                return val
            print(f"Debe ser al menos {min_val}.")
        except ValueError:
            print("Ingresa un número entero válido.")


def get_range(peticion: str, min_val: int = 1, max_val: int = 6) -> int:
    while True:
        try:
            p = int(input(peticion))
            if min_val <= p <= max_val:
                return p
            print(f"Debe estar entre {min_val} y {max_val}.")
        except ValueError:
            print("Debe ser un número entero.")



def select_file(title: str = "Seleccionar archivo", filetypes=(("Todos", "*.*"),)) -> str:
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def load_neural_network(file: str):
    from Neural_network.neural_network import Neural_network  
    try:
        with open(file, "r") as f:
            model_data = json.load(f)
        nn = Neural_network.from_dict(model_data)
        print("Red neuronal cargada exitosamente.")
        return nn
    except Exception as e:
        print(f"Error al cargar: {e}")
        return None


def save_neural_network(nn, file: str) -> None:
    try:
        with open(file, "w") as f:
            json.dump(nn.to_dict(), f, indent=4)
        print(f"Red neuronal guardada en {file}.")
    except Exception as e:
        print(f"Error al guardar: {e}")