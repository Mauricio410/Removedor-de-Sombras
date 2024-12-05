import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_shadows(image_path):
    # Leer la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de desenfoque
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Aplicar umbral adaptativo
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Invertir la imagen binaria
    binary = cv2.bitwise_not(binary)
    
    # Aplicar operación de cierre morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(image, image, mask=closed)
    
    return result

# Ejemplo de uso
result = remove_shadows('image.png')

# Mostrar la imagen usando matplotlib
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Resultado')
plt.axis('off')
plt.show()