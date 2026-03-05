# Red Neuronal desde Cero — Guía de Uso

Proyecto de red neuronal implementada con NumPy puro para clasificación de imágenes.

---

## Requisitos

```bash
pip install numpy matplotlib
```

---

## Cómo Ejecutar

```bash
python main.py
```

Al iniciar, se abrirá un diálogo para seleccionar el dataset (`.npz`). Luego aparece el menú principal.

---

## 📂 Formato del Dataset

El archivo `.npz` debe contener dos arrays:

| Clave     | Tipo       | Shape esperado         | Descripción              |
|-----------|------------|------------------------|--------------------------|
| `images`  | `uint8`    | `(N, H, W)`            | Imágenes en escala 0–255 |
| `labels`  | `int`      | `(N,)`                 | Etiquetas de clase       |

**Ejemplo para crearlo:**
```python
import numpy as np
np.savez("dataset.npz", images=mis_imagenes, labels=mis_etiquetas)
```

---

## 🖥️ Menú Principal

```
========== Menú Principal ==========
1. Cargar/Recargar dataset
2. Crear red neuronal
3. Cargar red neuronal (JSON)
4. Guardar red neuronal (JSON)
5. Entrenar          ← (no implementado aún)
6. Predecir
7. Ver errores de predicción
8. Salir
```

---

## Opciones del Menú

### 1. Cargar Dataset
Abre un explorador de archivos para seleccionar un `.npz`. Las imágenes se normalizan automáticamente dividiendo entre 255.

### 2. Crear Red Neuronal
Configura la arquitectura manualmente:
- Se pide el **número de capas** (mínimo 2).
- Para cada capa oculta: número de neuronas y activación (ReLU).
- La **última capa** se configura automáticamente con Softmax y el número de clases del dataset.

**Ejemplo de configuración:**
```
Capas: 3
Capa 1 → 128 neuronas, ReLU
Capa 2 → 64 neuronas,  ReLU
Capa 3 → 10 neuronas,  Softmax  ← automático
```

### 3. Cargar Red desde JSON
Carga un modelo previamente guardado. El archivo debe seguir el formato de `to_dict()`.

**Estructura del JSON:**
```json
{
  "input_shape": 784,
  "preprocess": { "scale": 255.0 },
  "layers": [
    {
      "type": "dense",
      "units": 128,
      "activation": "relu",
      "W": [[...]],
      "b": [[...]]
    },
    {
      "type": "dense",
      "units": 10,
      "activation": "softmax",
      "W": [[...]],
      "b": [[...]]
    }
  ]
}
```

### 4. Guardar Red en JSON
Guarda los pesos y arquitectura actual en un archivo `.json`. Se pide el nombre del archivo.

```
Nombre del archivo a guardar: mi_modelo.json
```

### 5. Entrenar
> ⚠️ No implementado en la versión actual.

### 6. Predecir
Corre el modelo sobre todo el dataset cargado e imprime:
- Las primeras 20 predicciones vs etiquetas reales.
- El **accuracy** sobre todo el dataset.

```
Predicciones:     [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
Etiquetas reales: [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
Precisión actual: 97.43%
```

### 7. Ver Errores de Predicción
Muestra una grilla de imágenes donde el modelo se equivocó, con la etiqueta real y la predicción.

```
¿Cuántos errores mostrar? (max recomendado 10): 10
```

---
## 📌 Notas

- El entrenamiento no está implementado; los modelos deben **cargarse desde JSON** con pesos pre-entrenados.
- La red aplana automáticamente las imágenes al momento de cargar el dataset.
- Los pesos se inicializan con `0.01 * np.random.randn(...)` y los biases en c

## 🏗️ Arquitectura Interna

```
Input (784)
    ↓
DenseLayer  →  dot(X, W) + b
    ↓
ReLU        →  max(0, x)
    ↓
DenseLayer  →  dot(X, W) + b
    ↓
Softmax     →  exp(x) / sum(exp(x))
    ↓
Output (10 probabilidades)
```

**Función de pérdida:** Categorical Cross-Entropy  
**Métrica:** Accuracy (argmax de probabilidades vs etiquetas reales)