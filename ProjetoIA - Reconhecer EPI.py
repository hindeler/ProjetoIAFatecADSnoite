import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# Carregar o modelo treinado
model = load_model('validador_epi3.h5')

# Inicializar o classificador de rostos pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a câmera
video_capture = cv2.VideoCapture(0)

# Função para processar a imagem
def process_frame(face):
    # Redimensionar o rosto detectado para o tamanho aceito pelo modelo
    img_resized = cv2.resize(face, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0)  # Expandir as dimensões para (1, 128, 128, 3)
    img_array = img_array / 255.0  # Normalizar a imagem

    # Fazer a predição usando o modelo
    prediction = model.predict(img_array)[0][0]  # Obter a predição como um único valor
    
    # Retornar se o EPI foi detectado (com base no valor da predição)
    if prediction > 0.7:
        return "Com Capacete", (0, 255, 0)  # Verde, com EPI
    elif prediction >= 0.4:
        return "Sem Capacete", (0, 0, 255)  # Vermelho, sem EPI
    else:
        return None, None  # Não mostrar nada

# Loop para capturar os frames da câmera e fazer a detecção
while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Erro ao capturar imagem.")
        break

    # Converter a imagem para escala de cinza para a detecção de rostos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Para cada rosto detectado, processar e classificar EPI
    for (x, y, w, h) in faces:
        # Recortar a região do rosto
        face = frame[y:y+h, x:x+w]
        
        # Chamar a função para detectar EPI
        label, color = process_frame(face)
        
        # Se o label não for None, desenhar o retângulo e mostrar o texto
        if label is not None:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Mostrar a imagem com o retângulo (se aplicável)
    cv2.imshow('Detecção de EPI', frame)

    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar a câmera e janelas
video_capture.release()
cv2.destroyAllWindows()
