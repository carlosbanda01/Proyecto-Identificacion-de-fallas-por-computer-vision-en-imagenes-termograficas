import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pywt
import joblib

def otsu(img):
    imagen_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    umbral_otsu, imagen_binaria = cv2.threshold(imagen_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result_image = cv2.bitwise_and(img, img, mask=imagen_binaria)
    return result_image, imagen_binaria

def HSI(img):
    imagen, mask = otsu(img)
    imagenes_I = []
    
    # Convertir imagen a RGB y luego a float en el rango [0, 1]
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) / 255.0

    # Descomposición de los canales de la imagen
    R, G, B = cv2.split(imagen)

    # Calcular el canal de Intensity
    I = (R + G + B) / 3
    I = (I * 255).astype(np.uint8)  # Escalar a 0-255 para visualizar
    I = cv2.GaussianBlur(I, (5, 5), 0)

    # Calcular el canal de Saturation
    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6) * min_RGB)
    S = (S * 255).astype(np.uint8)  # Escalar a 0-255 para visualizar

    # Calcular el canal de Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(num / (den + 1e-6))

    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)  # Normalizar el rango de H entre [0, 1]
    H = (H * 255).astype(np.uint8)  # Escalar a 0-255 para visualizar


    return I, mask


def resize(img, size=(224, 224)):
    img_I, mask = HSI(img)
    result_image_rez = cv2.resize(img_I, size)
    result_mask_rez = cv2.resize(mask, size)
    return result_image_rez, result_mask_rez



def clahe(img):
    img_rez, mask = resize(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # Si la imagen es en color, convertir a escala de grises
    if img_rez.ndim == 3:
        image = cv2.cvtColor(img_rez, cv2.COLOR_BGR2GRAY)
    else:
        image = img_rez

    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_clahe = clahe.apply(image_uint8)
    return image_clahe, mask

def normalizar_0_1(subbanda):
    return (subbanda - np.min(subbanda)) / (np.max(subbanda) - np.min(subbanda))

def extraer_subbandas_wavelet(image, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]
    cA = normalizar_0_1(np.array(cA))
    cH = normalizar_0_1(np.array(cH))
    cV = normalizar_0_1(np.array(cV))
    cD = normalizar_0_1(np.array(cD))
    return cA, cH, cV, cD



def extraer_caracteristicas_glcm(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return contrast, homogeneity


def pre(img):
    img_clahe, mask = clahe(img)
    GLCM_matrix=[]
    cA, cH, cV, cD = extraer_subbandas_wavelet(img_clahe)
    matriz_subbandas = [cA, cH, cV, cD]
    contrast,homogeneity = extraer_caracteristicas_glcm(img_clahe)
    GLCM_matrix.append([contrast,homogeneity])
    return matriz_subbandas, GLCM_matrix

def aplanar_y_concatenar_subbandas(subbandas):
    cA, cH, cV, cD = subbandas
    # Aplanar cada subbanda
    cA_flat = cA.flatten()
    cH_flat = cH.flatten()
    cV_flat = cV.flatten()
    cD_flat = cD.flatten()
    # Concatenar todas las subbandas aplanadas
    return np.concatenate([cA_flat, cH_flat, cV_flat, cD_flat]).reshape(1,-1)

def estimar_temp(imagen, limite_inferior, limite_superior):
    
    # Convertir a escala de grises si la imagen está en color
    if len(imagen.shape) == 3:  # Si tiene 3 canales, está en color
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen

    # Encontrar el valor máximo de intensidad en la imagen en escala de grises y su posición
    valor_maximo_pixel = np.max(imagen_gris)
    y_max, x_max = np.unravel_index(np.argmax(imagen_gris), imagen_gris.shape)

    # Mapear el valor máximo al rango de temperaturas
    temperatura_maxima = limite_inferior + (valor_maximo_pixel / 255) * (limite_superior - limite_inferior)

    # Convertir a BGR para poder dibujar en color (si estaba en escala de grises)
    if len(imagen.shape) == 2:
        imagen_marcada = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        imagen_marcada = imagen.copy()

    # Dibujar un círculo en el punto de mayor temperatura
    cv2.circle(imagen_marcada, (x_max, y_max), radius=10, color=(0, 0, 255), thickness=2)

    # Opcional: agregar el valor de la temperatura máxima en texto junto al marcador
    texto = f"{temperatura_maxima:.1f} °C"
    cv2.putText(imagen_marcada, texto, (x_max + 15, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    return imagen_marcada, temperatura_maxima