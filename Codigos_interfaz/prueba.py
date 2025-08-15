import os
import cv2
import ultralytics
print(cv2.__version__)
from ultralytics import YOLO
print(ultralytics.__version__)
# Cargar el modelo YOLO
modelo = YOLO("Yolov5s_fluke_imgs_best.pt")  # Asegúrate de que esta ruta sea correcta
ruta_imgs = "DATA_FLUKE"  # Carpeta que contiene las imágenes
ruta_output = "RESULTADOS_FLUKE"  # Carpeta donde guardar los resultados

# Crear la carpeta de salida si no existe
if not os.path.exists(ruta_output):
    os.makedirs(ruta_output)

# Extensiones permitidas de archivos de imagen
extensiones_permitidas = ['.jpg', '.jpeg', '.png', '.bmp']

# Obtener todas las imágenes en la carpeta con extensiones permitidas
imagenes = [f for f in os.listdir(ruta_imgs) if os.path.splitext(f)[1].lower() in extensiones_permitidas]

# Iterar sobre las imágenes y realizar inferencia
for imagen in imagenes:
    # Ruta completa de la imagen
    ruta_imagen = os.path.join(ruta_imgs, imagen)
    
    # Leer la imagen
    img = cv2.imread(ruta_imagen)

    # Verificar que la imagen se haya leído correctamente
    if img is None:
        print(f"No se pudo cargar la imagen {ruta_imagen}")
        continue

    # Realizar la inferencia con el modelo YOLO
    resultados = modelo(img, conf=0.7, iou=0.5)

    # Dibujar las cajas delimitadoras (bounding boxes) sobre la imagen
    resultado_img = resultados[0].plot()  # Esto dibuja las detecciones en la imagen

    # Guardar la imagen con los resultados en la carpeta de salida
    ruta_resultado = os.path.join(ruta_output, f"resultado_{imagen}")
    cv2.imwrite(ruta_resultado, resultado_img)

    # Mensaje de confirmación
    print(f"Resultado guardado en: {ruta_resultado}")
