import glob
import os
import cv2
import time
from emailing import send_email  # Importar la función send_email del módulo emailing
from threading import Thread  # Importar la clase Thread del módulo threading

# Inicialización de la cámara de video (0 representa la cámara predeterminada)
video = cv2.VideoCapture(0)
time.sleep(0)  # Pequeña pausa para permitir que la cámara se inicialice

# Variable para almacenar el primer fotograma (imagen) capturado
first_frame = None
status_list = []  # Lista para almacenar el estado de detección en cada iteración
count = 1  # Contador para nombrar las imágenes capturadas


# Función para limpiar la carpeta de imágenes
def clean_folder():
    print("clean function started")
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)
    print("clean function ended")


# Bucle principal que se ejecuta hasta que se presione la tecla "q"
while True:
    status = 0  # Estado de detección, 0 = no se detecta ningún cambio
    # Capturar un fotograma de la cámara
    check, frame = video.read()

    # Convertir el fotograma a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro Gaussiano para suavizar el fotograma en escala de grises
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Si es el primer fotograma, almacenarlo como referencia
    if first_frame is None:
        first_frame = gray_frame_gau

    # Calcular la diferencia absoluta entre el primer fotograma y el fotograma actual
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    # Aplicar umbralización para convertir la diferencia en una imagen binaria
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]

    # Aplicar dilatación a la imagen binaria para resaltar las áreas de cambio
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Mostrar la imagen binaria en una ventana llamada "My video"
    cv2.imshow("My video", thresh_frame)

    # Encontrar contornos en la imagen dilatada
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Recorrer los contornos encontrados
    for contour in contours:
        # Si el área del contorno es menor que 5000, se ignora
        if cv2.contourArea(contour) < 5000:
            continue
        # Obtener las coordenadas y dimensiones del rectángulo que rodea el contorno
        x, y, w, h = cv2.boundingRect(contour)
        # Dibujar un rectángulo verde alrededor del área de cambio en el fotograma original
        rectangle = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
        if rectangle.any():
            status = 1  # Cambiar el estado a 1 si se detecta un objeto
            cv2.imwrite(f"images/{count}image.png", frame)
            count = count + 1
            all_images = glob.glob("images/*.png")
            index = int(len(all_images) / 2)
            image_with_object = (all_images[index])  # Obtener la imagen con el objeto detectado

    status_list.append(status)  # Agregar el estado actual a la lista
    status_list = status_list[-2:]  # Mantener solo los dos últimos estados en la lista

    if status_list[0] == 1 and status_list[1] == 0:
        email_thread = Thread(target=send_email, args=(image_with_object,))
        email_thread.daemon = True
        clean_thread = Thread(target=clean_folder)
        clean_thread.daemon = True

        email_thread.start()  # Iniciar un hilo para enviar el correo electrónico
        clean_thread.start()  # Iniciar un hilo para limpiar la carpeta de imágenes

    # Mostrar el fotograma original con los rectángulos dibujados
    cv2.imshow("Video", frame)

    # Esperar una tecla presionada durante 1 milisegundo
    key = cv2.waitKey(1)

    # Si la tecla presionada es "q", salir del bucle
    if key == ord("q"):
        break

# Liberar la cámara y cerrar las ventanas abiertas
video.release()
cv2.destroyAllWindows()
