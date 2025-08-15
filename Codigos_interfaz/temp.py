import cv2
import numpy as np
import matplotlib.pyplot as plt

def convertir_a_HSI(imagen):
    # Convertir imagen de BGR a float para trabajar en el rango [0, 1]
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) / 255.0

    # Descomposición de los canales de la imagen
    R, G, B = cv2.split(imagen)
    
    # Calcular el canal de Intensity
    I = (R + G + B) / 3

    # Calcular el canal de Saturation
    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6) * min_RGB)

    # Calcular el canal de Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(num / (den + 1e-6))

    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)  # Normalizar el rango de H entre [0, 1]

    return H, S, I

# Cargar la imagen y convertirla a HSI
imagen = cv2.imread("DATA_FLUKE/INF1657011765269421690.png")
H, S, I = convertir_a_HSI(imagen)

# Mostrar los canales de H, S e I usando matplotlib
plt.figure(figsize=(12, 4))

# Canal H (Hue)
plt.subplot(1, 3, 1)
plt.imshow(H, cmap="hsv")
plt.colorbar()
plt.title("Canal H (Hue)")

# Canal S (Saturation)
plt.subplot(1, 3, 2)
plt.imshow(S, cmap="gray")
plt.colorbar()
plt.title("Canal S (Saturation)")

# Canal I (Intensity)
plt.subplot(1, 3, 3)
plt.imshow(I, cmap="gray")
plt.colorbar()
plt.title("Canal I (Intensity)")

plt.tight_layout()
plt.show()
