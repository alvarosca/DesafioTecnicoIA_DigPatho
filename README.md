# Desafío Técnico IA - DigPatho

Este proyecto crea y entrena un modelo de clasificación de imágenes utilizando la base de datos **FashionMNIST** y las arquitecturas **ResNet** y **EfficientNet**. El objetivo principal es entrenar un modelo eficiente, almacenar los pesos óptimos y generar un archivo `submission.csv` para la participación en el Hackathon "Identify the Apparels" organizado por Analytics Vidhya.

---

## Funcionamiento

### Evaluación de un modelo entrenado

Para evaluar un modelo previamente entrenado, ejecuta uno de los siguientes comandos según la arquitectura deseada:

```bash
python main.py -test -arch ResNet --ckpt_file ./checkpoint/resnet_ckpt.pth
```

```bash
python main.py -test -arch EfficientNet --ckpt_file ./checkpoint/efficientnet_ckpt.pth
```

Estos comandos generan predicciones para el conjunto de prueba y guardan el archivo `submission.csv` en la carpeta `submission/`.

### Rendimiento de los Modelos Pre-entrenados

| Modelo          | Dataset          | Precisión |  Params. |
|-----------------|------------------|----------|-------------------| 
| **ResNet**      | FashionMNIST     | 95.17%   |  272k  | 
| **EfficientNet**| FashionMNIST     | 94.18%   |  333k  |

### Entrenamiento de un modelo desde cero

Para iniciar el entrenamiento desde cero, especifica la arquitectura y el número de 'epochs':

```bash
python main.py --architecture ResNet --epochs 300 --ckpt_file ./checkpoint/new_resnet.pth
```

### Visualización de la arquitectura del modelo

Agrega la opción `--summary` para mostrar un resumen detallado de la arquitectura utilizada:

```bash
python main.py --architecture ResNet --summary
```

Ejemplo de salida:

```bash
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
ResNet                                   --
├─Conv2d: 1-1                            144
├─BatchNorm2d: 1-2                       32
├─ReLU: 1-3                              --
├─Sequential: 1-4                        --
...
│    └─BasicBlock: 2-9                   --
│    │    └─Conv2d: 3-43                 36,864
│    │    └─BatchNorm2d: 3-44            128
│    │    └─ReLU: 3-45                   --
│    │    └─Conv2d: 3-46                 36,864
│    │    └─BatchNorm2d: 3-47            128
├─AvgPool2d: 1-7                         --
├─Linear: 1-8                            650
=================================================================
Total params: 272,186
Trainable params: 272,186
Non-trainable params: 0
=================================================================
```

---

### Opciones principales

| Argumento                      | Descripción                                                   | Valor por defecto       |
|--------------------------------|---------------------------------------------------------------|-------------------------|
| `--resume`, `-r`               | Reanuda el entrenamiento desde un checkpoint                  | `False`                 |
| `--summary`                    | Muestra un resumen de la arquitectura del modelo              | `False`                 |
| `--dont_display_progress_bar`  | Oculta la barra de progreso                                   | `False`                 |
| `--test_model`, `-test`        | Evalúa el modelo utilizando un checkpoint                     | `False`                 |
| `--architecture`, `-arch`      | Especifica la arquitectura a utilizar (`ResNet`, `EfficientNet`) | `ResNet`              |
| `--batch_size`                 | Define el tamaño del batch para el entrenamiento              | `64`                    |
| `--epochs`                     | Establece el número de épocas para el entrenamiento           | `300`                   |
| `--lr`, `--learning-rate`      | Tasa de aprendizaje del optimizador                           | `0.1`                   |
| `--p`                          | Probabilidad de borrado aleatorio para `Random Erasing`        | `0.5`                   |

---
