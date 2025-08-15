import sys
import matplotlib.pyplot as plt
import io
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QLabel, QMessageBox, QDialog, QPushButton, QVBoxLayout, QTextEdit
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QRect, QDateTime, QBuffer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from io import BytesIO
from PIL import Image

import ultralytics
from ultralytics import YOLO

import joblib
from pre import pre, otsu, resize, clahe, aplanar_y_concatenar_subbandas, estimar_temp

class SeleccionRectanguloDialog(QDialog):
    def __init__(self, ruta_imagen, parent=None):
        super().__init__(parent)

        self.ruta_imagen = ruta_imagen
        self.seleccion_iniciada = False
        self.punto_inicial = None
        self.punto_final = None
        self.rectangulo_confirmado = None

        # Cargar la imagen original en el QLabel
        self.imagen_label = QLabel(self)
        img_original1=cv2.imread(self.ruta_imagen)
        img_original = cv2.resize(img_original1, (640, 512), interpolation=cv2.INTER_AREA)
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        h, w, ch = img_original_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_original_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Guardar el QPixmap de la imagen para redibujar sobre él
        self.pixmap_original = QPixmap.fromImage(q_img)
        self.imagen_label.setPixmap(self.pixmap_original)

        # Crear un botón de confirmación
        self.boton_confirmar = QPushButton("Confirmar selección", self)
        self.boton_confirmar.clicked.connect(self.confirmar_seleccion)

        # Layout para organizar el QLabel y el botón
        layout = QVBoxLayout()
        layout.addWidget(self.imagen_label)
        layout.addWidget(self.boton_confirmar)
        self.setLayout(layout)

        self.setWindowTitle("Seleccionar área del motor")

    def confirmar_seleccion(self):
        # Al confirmar, guardamos el rectángulo seleccionado
        if self.punto_inicial and self.punto_final:
            self.rectangulo_confirmado = QRect(self.punto_inicial, self.punto_final)
            self.accept()  # Cerrar el diálogo y devolver la selección

    def mousePressEvent(self, event):
        # Iniciar la selección del rectángulo
        if event.button() == Qt.LeftButton:
            self.seleccion_iniciada = True
            self.punto_inicial = event.pos()
            self.punto_final = self.punto_inicial
            self.update_rectangulo()  # Redibujar el rectángulo en tiempo real

    def mouseMoveEvent(self, event):
        # Actualizar la selección mientras se arrastra el mouse
        if self.seleccion_iniciada:
            self.punto_final = event.pos()
            self.update_rectangulo()  # Redibujar el rectángulo en tiempo real

    def mouseReleaseEvent(self, event):
        # Finalizar la selección cuando se suelta el botón del mouse
        if event.button() == Qt.LeftButton:
            self.seleccion_iniciada = False
            self.punto_final = event.pos()
            self.update_rectangulo()  # Redibujar el rectángulo final

    def update_rectangulo(self):
        # Redibujar el pixmap original y superponer el rectángulo seleccionado
        pixmap_temp = QPixmap(self.pixmap_original)  # Crear una copia del pixmap original para redibujar
        painter = QPainter(pixmap_temp)

        if self.punto_inicial and self.punto_final:
            # Crear el rectángulo con las posiciones del mouse
            rect = QRect(self.punto_inicial, self.punto_final)
            pen = QPen(Qt.blue, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)

        painter.end()
        self.imagen_label.setPixmap(pixmap_temp)  # Actualizar el QLabel con la imagen modificada

class ConfigWindow(QDialog):
    def __init__(self, parent=None):
        super(ConfigWindow, self).__init__(parent)
        uic.loadUi('config_window.ui', self)  # Cargar el diseño .ui de la ventana de configuración
        self.setWindowTitle("Configuraciones")

        # Variables para almacenar los límites de temperatura
        self.limite_inferior = None
        self.limite_superior = None

        # Conectar los botones
        self.bt_acp.clicked.connect(self.guardar_y_cerrar)

    def guardar_y_cerrar(self):
        # Guardar los valores de temperatura de los campos min_temp_box y max_temp_box
        self.limite_inferior = float(self.min_temp_box.value())
        self.limite_superior = float(self.max_temp_box.value())
        
        # Cerrar el diálogo con estado "aceptado"
        self.accept()

    def obtener_configuracion(self):
        return self.limite_inferior, self.limite_superior
    

class ObservacionesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Añadir Observaciones")
        self.setGeometry(100, 100, 400, 300)
        
        # Campo de texto para las observaciones
        self.text_edit = QTextEdit(self)
        
        # Botón para aceptar
        self.btn_aceptar = QPushButton("Aceptar", self)
        self.btn_aceptar.clicked.connect(self.accept)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.btn_aceptar)
        self.setLayout(layout)
    
    def get_observaciones(self):
        return self.text_edit.toPlainText()

#___________________________________________________________________________________________________________________________
class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()

        uic.loadUi("interfaz_COD.ui", self)  # Carga el archivo .ui
        
        self.showMaximized()
        # Conectar el botón para seleccionar la imagen
        self.bt_img.clicked.connect(self.seleccionar_imagen)
        
        # Conectar el botón para hacer la inferencia
        self.bt_id.clicked.connect(self.realizar_inferencia)

        self.bt_open_config.clicked.connect(self.abrir_ventana_configuracion)

        self.bt_pred.clicked.connect(self.predecir)

        self.bt_rst.clicked.connect(self.limpiar_contenido)

        self.bt_save.clicked.connect(self.generar_reporte_pdf)
        
        # Variable para almacenar la ruta de la imagen seleccionada
        self.ruta_imagen = None
        self.ruta_logo = "logoUdep.png"
        self.limite_inferior = 20  # Valor por defecto
        self.limite_superior = 80

        # Cargar el modelo YOLO
        self.modelo_yolo = YOLO("YOLOv5m_FULL.pt")
        
        self.texto_h = (
            "Mantén la ventilación libre de obstrucciones: Asegúrate de que las entradas y salidas de aire del motor "
            "estén limpias y sin bloqueos para prevenir sobrecalentamiento. "
            "Evita sobrecargar el motor: Asegúrate de que el motor opere dentro de su capacidad nominal para prolongar su vida útil. "
            "Comprueba las conexiones eléctricas: Revisa visualmente las conexiones para asegurarte de que estén bien apretadas, "
            "evitando puntos calientes y arcos eléctricos."
        )

        self.texto_m = (
            "Instala soportes y anclajes sólidos: Asegúrate de que el motor esté firmemente "
            "montado sobre una base estable para reducir el riesgo de desalineación por vibración."
            "Inspecciona visualmente las juntas y bases: Verifica si hay desgaste visible o tornillos flojos" 
            "en las juntas o en la base, lo cual podría indicar desalineación o movimiento."
        )

        self.texto_r = (
            "Evita ciclos de arranque y parada frecuentes: Los ciclos de arranque frecuentes pueden incrementar "
            "el estrés en las barras del rotor."
            "Monitorea la temperatura de operación: Mantén el motor en un ambiente adecuado y, si es posible, "
            "utiliza ventilación externa para evitar temperaturas que pudieran agravar el daño en las barras."
            "Evita el arranque en condiciones de carga máxima: Intenta arrancar el motor con poca carga para "
            "minimizar el esfuerzo en el rotor durante el inicio."
        )




        
        # Mostrar la ventana
        self.show()

    def seleccionar_imagen(self):
        # Abrir cuadro de diálogo para seleccionar una imagen
        opciones = QFileDialog.Options()
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "",
                                                 "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)", options=opciones)

        # Si se selecciona un archivo
        if archivo:
            self.ruta_imagen = archivo
            pixmap = QPixmap(archivo)
            
            # Mostrar la imagen seleccionada en el QLabel
            self.imagen_label.setPixmap(pixmap.scaled(self.imagen_label.size(), Qt.KeepAspectRatio))

    def realizar_inferencia(self):
        if self.ruta_imagen:
            # Cargar la imagen seleccionada
            img = cv2.imread(self.ruta_imagen)
            #img=cv2.resize(img1,(640,512),interpolation=cv2.INTER_AREA)
            
            # Realizar la inferencia con el modelo YOLO
            resultados = self.modelo_yolo.predict(source=img, save=False, iou=0.8, conf=0.6, imgsz=320)

            if resultados[0].boxes is not None and len(resultados[0].boxes) > 0:
                # Dibujar las cajas delimitadoras (bounding boxes) sobre la imagen
                resultado_img = resultados[0].plot()  # Esto dibuja las detecciones en la imagen

                # Convertir la imagen de BGR (OpenCV) a RGB (para PyQt5)
                resultado_img_rgb = cv2.cvtColor(resultado_img, cv2.COLOR_BGR2RGB)

                # Convertir la imagen de OpenCV a QImage para mostrarla en QLabel
                h, w, ch = resultado_img_rgb.shape
                bytes_per_line = ch * w
                q_img = QPixmap(QImage(resultado_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

                # Mostrar la imagen con las detecciones en el QLabel
                self.imagen_label.setPixmap(q_img.scaled(self.imagen_label.size(), Qt.KeepAspectRatio))

                # Obtener el primer rectángulo detectado por YOLO y recortarlo
                box = resultados[0].boxes.xyxy[0]  # Suponiendo que sólo necesitas el primer rectángulo
                rect = QRect(int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1]))
                print("Rectangulo YOLO")
                print(rect)
                print(int(box[0]))
                self.recortar_rectangulo(rect)
            
                print(ultralytics.__version__)
            self.mostrar_confirmacion()

    def mostrar_confirmacion(self):
        # Crear un cuadro de mensaje
        mensaje = QMessageBox()
        mensaje.setWindowTitle("Confirmación de motor")
        mensaje.setText("¿Es correcto el motor identificado?")
        
        # Añadir botones
        mensaje.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        mensaje.setDefaultButton(QMessageBox.Yes)
        
        # Personalizar el texto de los botones
        boton_si = mensaje.button(QMessageBox.Yes)
        boton_si.setText("Sí, usar ese")
        
        boton_no = mensaje.button(QMessageBox.No)
        boton_no.setText("No, seleccionar manualmente")
        
        # Mostrar el cuadro de mensaje y capturar la respuesta
        respuesta = mensaje.exec_()
        
        if respuesta == QMessageBox.Yes:
            print("El usuario seleccionó 'Sí, usar ese'")
        else:
            self.seleccionar_rectangulo_manual()

    def seleccionar_rectangulo_manual(self):
        # Crear y abrir el diálogo de selección manual del rectángulo
        dialogo = SeleccionRectanguloDialog(self.ruta_imagen, self)
        img = cv2.imread(self.ruta_imagen)
        alto, ancho, canales = img.shape
        if dialogo.exec_() == QDialog.Accepted:
            # Si el usuario confirma la selección, guardamos el rectángulo
            self.rectangulo_seleccionado = dialogo.rectangulo_confirmado
            x1 = self.rectangulo_seleccionado.left()
            x1r=x1*ancho/640
            y1 = self.rectangulo_seleccionado.top()
            y1r=y1*alto/512
            x2 = self.rectangulo_seleccionado.right()
            x2r=x2*ancho/640
            y2 = self.rectangulo_seleccionado.bottom()
            y2r=y2*alto/512
            print("COORDENADAS MANUALES")
            print(x1)
            print(y1)
            print(x2)
            print(y2)
            print("COORDENADAS MANUALES reales")
            print(x1r)
            print(y1r)
            print(x2r-x1r)
            print(y2r-y1r)
            self.mostrar_imagen_con_rectangulo()
            self.rectangulo_seleccionado_real=QRect(round(x1r),round(y1r),round(x2r-x1r),round(y2r-y1r))
            self.recortar_rectangulo(self.rectangulo_seleccionado_real)


    def mostrar_imagen_con_rectangulo(self):
        # Mostrar la imagen con el rectángulo seleccionado
        img_original = cv2.imread(self.ruta_imagen)
        #img_original=cv2.resize(img_original1, (640,512), interpolation=cv2.INTER_AREA)
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        h, w, ch = img_original_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_original_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Convertir a pixmap
        pixmap = QPixmap.fromImage(q_img)
        painter = QPainter(pixmap)
        img = cv2.imread(self.ruta_imagen)
        alto, ancho, canales = img.shape
        # Dibujar el rectángulo confirmado
        if self.rectangulo_seleccionado:
            pen = QPen(Qt.blue, 2, Qt.SolidLine)
            painter.setPen(pen)
            x1 = self.rectangulo_seleccionado.left()
            x1r=x1*ancho/640
            y1 = self.rectangulo_seleccionado.top()
            y1r=y1*alto/512
            x2 = self.rectangulo_seleccionado.right()
            x2r=x2*ancho/640
            y2 = self.rectangulo_seleccionado.bottom()
            y2r=y2*alto/512
            self.rectangulo_seleccionado_real=QRect(round(x1r),round(y1r),round(x2r-x1r),round(y2r-y1r))
            painter.drawRect(self.rectangulo_seleccionado_real)

        painter.end()

        # Mostrar la imagen en el QLabel principal
        self.imagen_label.setPixmap(pixmap.scaled(self.imagen_label.size(), Qt.KeepAspectRatio))

    def recortar_rectangulo(self, rect):
        # Cargar la imagen original redimensionada a 640x512
        img_original = cv2.imread(self.ruta_imagen)
        #img_original = cv2.resize(img_original1, (640, 512), interpolation=cv2.INTER_AREA)

        # Convertir el QRect a coordenadas de la imagen
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        # Recortar el área del rectángulo
        global img_recortada
        img_recortada = img_original[y1:y2, x1:x2]
        img_rec=cv2.resize(img_recortada,(224,224))

        # Convertir la imagen recortada a RGB para PyQt5
        img_recortada_rgb = cv2.cvtColor(img_rec, cv2.COLOR_BGR2RGB)

        # Convertir la imagen recortada a QImage
        h, w, ch = img_recortada_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_recortada_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Mostrar la imagen recortada en el QLabel img_mostrar
        self.img_mostrar.setPixmap(QPixmap.fromImage(q_img).scaled(self.img_mostrar.size(), Qt.KeepAspectRatio))

        global matriz_subbandas, glcm
        matriz_subbandas, glcm = pre(img_rec)  # Aquí llamas a la función preprocesamiento

        # Suponiendo que quieras mostrar la primera subbanda como resultado del preprocesamiento (puedes cambiar esto)
        img_preprocesada = matriz_subbandas[1]  # cA, o cualquiera de las subbandas

        # Convertir la imagen preprocesada a uint8 para mostrarla
        img_preprocesada_uint8 = (img_preprocesada * 255).astype(np.uint8)

        # Convertir la imagen preprocesada a RGB (si es necesario) para PyQt5
        img_preprocesada_rgb = cv2.cvtColor(img_preprocesada_uint8, cv2.COLOR_GRAY2RGB)

        # Convertir la imagen preprocesada a QImage
        h, w, ch = img_preprocesada_rgb.shape
        bytes_per_line = ch * w
        q_img_pre = QImage(img_preprocesada_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Mostrar la imagen preprocesada en el QLabel img_mostrar
        self.histogram_img.setPixmap(QPixmap.fromImage(q_img_pre).scaled(self.img_mostrar.size(), Qt.KeepAspectRatio))

    def generar_grafico_radar(self, probabilidades, clases, label_widget: QLabel):
        
        # Asegurarse de que el polígono esté cerrado
        probabilidades = list(probabilidades) + [probabilidades[0]]
        num_clases = len(clases)
        angulos = np.linspace(0, 2 * np.pi, num_clases, endpoint=False).tolist()
        angulos += angulos[:1]  # Cerrar el polígono

        # Crear la figura de radar
        fig, ax = plt.subplots(figsize=(4.5,4.5), subplot_kw=dict(polar=True))
        ax.plot(angulos, probabilidades, color='b', linewidth=2, linestyle='solid')
        ax.fill(angulos, probabilidades, color='b', alpha=0.25)


        # Configurar etiquetas de los ejes
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(clases)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0, 1)


        # Guardar la figura en un buffer de memoria
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Cargar la imagen en el QLabel
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), 'PNG')
        global radar_pixmap
        label_widget.setPixmap(pixmap)
        radar_pixmap = pixmap
        
        # Cerrar el buffer y la figura para liberar memoria
        buf.close()
        plt.close(fig)
    
    def predecir (self):
        
        # Especifica las rutas de los archivos que guardaste
        ruta_svm = "SVM/SVM_50_componentes_C_1_gamma_0.01_full_prob.pkl"
        ruta_scaler = "SVM/scaler_50_componentes_C_1_gamma_0.01_full.pkl"
        ruta_pca = "SVM/pca_50_componentes_C_1_gamma_0.01_full.pkl"

        ruta_svm_m = "SVM_mis/SVM_MIS_50+2_componentes_C_0.5_gamma_0.01_full_prob.pkl"
        ruta_scaler_m = "SVM_mis/scaler_MIS_50+2_componentes_C_1_gamma_0.01_full.pkl"
        ruta_pca_m = "SVM_mis/pca_MIS_50+2_componentes_C_1_gamma_0.01_full.pkl"

        ruta_svm_r = "SVM_rot/SVM_rot_50+2_componentes_C_0.5_gamma_0.01_full_prob.pkl"
        ruta_scaler_r = "SVM_rot/scaler_rot_50+2_componentes_C_1_gamma_0.01_full.pkl"
        ruta_pca_r =  "SVM_rot/pca_rot_50+2_componentes_C_1_gamma_0.01_full.pkl"

        # Cargar el modelo SVM, el escalador y el PCA desde las rutas guardadas
        svm_model_50_2 = joblib.load(ruta_svm)
        scaler = joblib.load(ruta_scaler)
        pca = joblib.load(ruta_pca)

        svm_model_m = joblib.load(ruta_svm_m)
        scaler_m = joblib.load(ruta_scaler_m)
        pca_m = joblib.load(ruta_pca_m)

        svm_model_r = joblib.load(ruta_svm_r)
        scaler_r = joblib.load(ruta_scaler_r)
        pca_r = joblib.load(ruta_pca_r)

        # Suponiendo que tienes una nueva matriz de datos a la que quieres hacerle predicciones
        # Aquí 'matriz_nueva' es la matriz que deseas predecir (reemplaza por tu matriz real)
        # Debe tener el mismo formato que los datos de entrenamiento originales, antes de la transformación PCA
        matriz_nueva = aplanar_y_concatenar_subbandas(matriz_subbandas) # Reemplaza por tu propia matriz de entrada

        # Preprocesar la matriz (escalado y PCA)
        matriz_nueva_escalada = scaler.transform(matriz_nueva)
        matriz_nueva_pca = pca.transform(matriz_nueva_escalada)

        X = np.hstack((matriz_nueva_pca, glcm))

        # Hacer predicciones con el modelo SVM
        #prediccion = svm_model_50_2.predict(matriz_nueva_pca)
        #print(prediccion)
        global prediccion
        probabilidades = svm_model_50_2.predict_proba(X)[0]  # Convertir a porcentaje
        clases = svm_model_50_2.classes_
        prediccion = clases[probabilidades.argmax()]
        print(probabilidades)
        print(prediccion)

        if prediccion == 0:
            resultado = "Motor sano"
        elif prediccion == 1:

            proba_mis = svm_model_m.predict_proba(X)[0]
            clases_m = svm_model_m.classes_
            prediccion_m = clases_m[proba_mis.argmax()]

            if prediccion_m == 0:
                resultado = "Motor desalineado grado 1"
            elif prediccion_m == 1:
                resultado = "Motor desalineado grado 2"
            elif prediccion_m == 2:
                resultado = "Motor desalineado grado 3"

        elif prediccion == 2:

            proba_r = svm_model_r.predict_proba(X)[0]
            clases_r = svm_model_r.classes_
            prediccion_r = clases_r[proba_r.argmax()]

            if prediccion_r == 0:
                resultado = "1 barra de rotor rota"
            elif prediccion_r == 1:
                resultado = "3 barras de rotor rotas"
            elif prediccion_r == 2:
                resultado = "6 barras de rotor rotas"

        self.res_label.setText(resultado)

        global imagen_marcada, temperatura_maxima
        imagen_marcada, temperatura_maxima = estimar_temp(img_recortada, limite_inferior=self.limite_inferior,limite_superior=self.limite_superior)
            
         # Mostrar la imagen marcada en la interfaz
        #h, w, ch = imagen_marcada.shape
        #bytes_per_line = ch * w
        #q_img = QImage(imagen_marcada.data, w, h, bytes_per_line, QImage.Format_RGB888)
        #self.imagen_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.imagen_label.size(), Qt.KeepAspectRatio))
            
            # Mostrar la temperatura máxima estimada en el QLabel correspondiente
        self.temp_lbl.setText(f"{temperatura_maxima:.2f} °C")

        #probabilidades = [70, 20, 10]  # Reemplaza con tus datos reales
        clases_str = ["Healthy", "Misalignment", "Rotor bars broken"]
        
        # Llama a la función y pasa el QLabel en el que quieres mostrar el gráfico
        self.generar_grafico_radar(probabilidades, clases_str, self.label_grafico)
    
    def agregar_texto_ajustado(self, canvas, texto, x, y, ancho_maximo, altura_linea):
        """
        Función auxiliar para dividir el texto en líneas que se ajusten al ancho máximo especificado.
        """
        palabras = texto.split(' ')
        linea_actual = ""
        for palabra in palabras:
            prueba_linea = f"{linea_actual} {palabra}".strip()
            if canvas.stringWidth(prueba_linea, "Helvetica", 10) < ancho_maximo:
                linea_actual = prueba_linea
            else:
                canvas.drawString(x, y, linea_actual)
                y -= altura_linea
                linea_actual = palabra
        if linea_actual:
            canvas.drawString(x, y, linea_actual)
        return y  # Retorna la posición Y después de la última línea
    
    def generar_reporte_pdf(self):

        # Preguntar si desea añadir observaciones
        respuesta = QMessageBox.question(self, "Añadir Observaciones", "¿Desea añadir observaciones al reporte?", 
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        observaciones = ""
        
        # Si el usuario desea añadir observaciones, abrir la ventana emergente
        if respuesta == QMessageBox.Yes:
            dialogo_observaciones = ObservacionesDialog(self)
            if dialogo_observaciones.exec_() == QDialog.Accepted:
                observaciones = dialogo_observaciones.get_observaciones()

        # Pedir al usuario la ubicación y el nombre del archivo PDF
        opciones = QFileDialog.Options()
        archivo, _ = QFileDialog.getSaveFileName(self, "Guardar Reporte PDF", "", "PDF Files (*.pdf)", options=opciones)
        
        if archivo:
            # Crear el PDF
            c = canvas.Canvas(archivo, pagesize=A4)
            ancho, alto = A4
            alto = alto - 50

            if self.ruta_logo:
                c.drawImage(self.ruta_logo, 40, alto - 65 + 50, width=110, height=40)
            c.setFont("Helvetica", 15)
            c.drawString(380, alto - 50+50, "Grupo 10 COD")
            # Título
            c.setFont("Helvetica-Bold", 20)
            c.drawString(40, alto - 40, "Reporte de Detección de Fallas en Motores AC")

            # Fecha y hora
            fecha_hora = QDateTime.currentDateTime().toString("dd-MM-yyyy hh:mm:ss")
            c.setFont("Helvetica", 10)
            c.drawString(40, alto - 60, f"Fecha y hora de creación: {fecha_hora}")

            # Foto original
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, alto - 100, "Imagen Original:")
            if self.ruta_imagen:
                c.drawImage(self.ruta_imagen, 40, alto - 280, width=200, height=150)

            # Convertir las subbandas a imágenes en memoria y agregar al PDF
            c.drawString(280, alto - 100, "Preprocesamiento:")
            subbandas = ['cA', 'cH', 'cV', 'cD']
            x_offset = 280
            y_offset = alto - 200
            for i, subbanda in enumerate(matriz_subbandas):
                # Normalizar y convertir la subbanda a uint8
                subbanda_uint8 = (subbanda * 255).astype(np.uint8)
                subbanda_rgb = cv2.cvtColor(subbanda_uint8, cv2.COLOR_GRAY2RGB)
                
                # Convertir la subbanda en una imagen de `BytesIO`
                img_io = io.BytesIO()
                is_success, buffer = cv2.imencode(".png", subbanda_rgb)
                img_io.write(buffer)
                img_io.seek(0)
                
                # Insertar la imagen en el PDF
                c.setFont("Helvetica", 12)
                c.drawString(x_offset, y_offset+60, f"Subbanda {subbandas[i]}:")
                img_reader = ImageReader(img_io)
                c.drawImage(img_reader, x_offset, y_offset - 60, width=110, height=110)

                # Mover la posición para la próxima subbanda
                x_offset += 120
                if x_offset > 400:
                    x_offset = 280
                    y_offset -= 140
            
            
            # Foto con el punto de temperatura máxima marcado
            c.setFont("Helvetica-Bold", 12)
            max_temp_text = self.temp_lbl.text()
            c.drawString(40, alto - 300, f"Temperatura Máxima: {max_temp_text}")

            # Convertir la imagen marcada a PNG en memoria
            img_marcada_rgb = cv2.cvtColor(imagen_marcada, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_marcada_rgb)
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            # Agregar la imagen marcada al PDF usando ImageReader
            c.drawImage(ImageReader(img_buffer), 40, alto - 460, width=200, height=150)

            # Diagnóstico
            c.setFont("Helvetica-Bold", 12)
            c.drawString(280, alto - 640, "Diagnóstico:")
            c.setFont("Helvetica", 12)
            diagnostico_texto = self.res_label.text()
            c.drawString(280, alto - 660, diagnostico_texto)

            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, alto - 490, "Recomendaciones:")
            c.setFont("Helvetica", 9)
            if prediccion == 0:    
                self.agregar_texto_largo(c, self.texto_h, x_inicial=40, y_inicial=560, max_x=250, line_height=15)
            elif prediccion == 1:
                self.agregar_texto_largo(c, self.texto_m, x_inicial=40, y_inicial=560, max_x=250, line_height=15)
            elif prediccion ==2:
                self.agregar_texto_largo(c, self.texto_r, x_inicial=40, y_inicial=560, max_x=250, line_height=15)

            if 'radar_pixmap' in globals():
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                radar_pixmap.toImage().save(buffer, "PNG")
                png_data = buffer.data()
                img_buffer = BytesIO(png_data)

                # Crear un ImageReader desde el buffer y agregarlo al PDF
                image_reader = ImageReader(img_buffer)
                c.setFont("Helvetica-Bold", 12)
                c.drawString(280, alto - 430, "Gráfico de Probabilidades:")
                c.drawImage(image_reader, 280, alto - 630, width=200, height=200)

            # Observaciones
            if observaciones:
                if observaciones:
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(40, alto - 750+50, "Observaciones:")
                    c.setFont("Helvetica", 10)
                    y_final = self.agregar_texto_ajustado(c, observaciones, 40, alto - 770+50, ancho_maximo=500, altura_linea=12)
                    # `y_final` se puede usar para continuar con otro contenido, si es necesario.

            # Guardar y cerrar el PDF
            c.save()
            QMessageBox.information(self, "Reporte PDF", "Reporte generado y guardado con éxito.")

    def agregar_texto_largo(self, c, texto, x_inicial=40, y_inicial=600, max_x=250, line_height=15):
        
        # Variables iniciales
        x = x_inicial
        y = A4[1] - y_inicial  # y inicial tomando en cuenta la altura de la página
        palabras = texto.split()  # Dividir el texto en palabras

        # Recorrer cada palabra y agregarla con condiciones de salto de línea
        for palabra in palabras:
            palabra_ancho = c.stringWidth(palabra + " ", "Helvetica", 10)  # Ancho de la palabra
            if x + palabra_ancho > max_x:  # Si supera el límite en x, hacer un salto de línea
                x = x_inicial  # Reiniciar x
                y -= line_height  # Bajar a la siguiente línea
            
            # Dibujar palabra en la posición actual
            c.drawString(x, y, palabra)
            x += palabra_ancho  # Mover la posición x para la siguiente palabra


    def limpiar_contenido(self):
        """Función para limpiar el contenido de los QLabel al presionar bt_rst"""
        self.imagen_label.clear()  # Limpiar QLabel imagen_label
        self.img_mostrar.clear()   # Limpiar QLabel img_mostrar
        self.histogram_img.clear() # Limpiar QLabel histogram_img
        self.res_label.clear()     # Limpiar QLabel res_label
        self.temp_lbl.clear()
        self.label_grafico.clear()
        print("Contenido de los QLabel limpiado")


    def abrir_ventana_configuracion(self):
        # Crear una instancia de la ventana secundaria de configuración
        self.config_window = ConfigWindow(self)
        if self.config_window.exec_() == QDialog.Accepted:
            # Si la ventana de configuración se acepta, actualizar los límites de temperatura
            self.limite_inferior, self.limite_superior = self.config_window.obtener_configuracion()
            print("Límites de temperatura actualizados:", self.limite_inferior, self.limite_superior)


    


# Iniciar la aplicación PyQt
app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
