import os
import cv2
from ultralytics import YOLO
import numpy as np
import calib

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around  #Para diminuir o range mudar o hue - e hue
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

def get_team_detect(frame, cor_mandante, cor_visitante, cor_juiz):
    hsvImage = cv2.cvtColor(frame[int(ymin):int(ymax), int(xmin):int(xmax)], cv2.COLOR_BGR2HSV)
    #Mandante
    lowerLimit, upperLimit = get_limits(color=cor_mandante)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    count_mandante = cv2.countNonZero(mask)

    #Visitante
    lowerLimit, upperLimit = get_limits(color=cor_visitante)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    count_visitante = cv2.countNonZero(mask)

    #Juiz
    lowerLimit, upperLimit = get_limits(color=cor_juiz)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    count_juiz = cv2.countNonZero(mask)

    lista_cor = [cor_visitante, cor_mandante, cor_juiz]
    lista_count = [count_visitante, count_mandante, count_juiz]

    posicao_lista = lista_count.index(max(lista_count))
    cor_classif = lista_cor[posicao_lista] 

    return cor_classif

def get_team_apont(track_id, lista_mandante, lista_visitante, lista_juiz, cor_mandante, cor_visitante, cor_juiz):
    if track_id in lista_mandante:
        cor_classif = cor_mandante
    elif track_id in lista_visitante:
        cor_classif = cor_visitante
    elif track_id in lista_juiz:
        cor_classif = cor_juiz
    else: cor_classif = (0,0,0)

    return cor_classif

def plot_ground(frame_ground, matrix, xmin, ymin, xmax, ymax, cor_classif):
    x_p = (xmin + xmax) / 2
    y_p = ymax
    p_original = np.float32([[x_p,y_p, 1]])

    coor_transf = np.dot(matrix, p_original.T).T
    coor_transf = coor_transf / coor_transf[0, 2]

    p_trasf = (int(coor_transf[0][0]),int(coor_transf[0][1]))

    cv2.circle(frame_ground, p_trasf, 10, cor_classif,-1)

    return frame_ground


video_path = os.path.join('.', 'data', 'sport1.mp4')
ground_path = 'data/dst.jpg'

# Definir se vai pltoar os pontos ou passar a matriz diretamente
switch_matrix = 0

if switch_matrix == 0:
    matrix = calib.create_points(video_path, ground_path)
    print(matrix)
else: matrix = np.array([[-2.0589,-3.0571,2282.1],
                        [-0.198,-7.6683,3413],
                        [-0.0002981,-0.0069014,1]])

# Campo em 2D
ground = cv2.imread(ground_path, -1)

#Vídeo
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Modelo de detecção
model = YOLO("yolov8n.pt")

#Cores dos times, goleiro e juiz
cor_mandante = [40, 56, 244]
cor_visitante = [0, 192, 255]
cor_juiz = [0, 0, 0]

# Para o método de times por apontamento, pegar os IDs e escrever aqui
lista_mandante = [2,3,11,12,20]
lista_visitante = [1,4,5,6,9,7,10,14]
lista_juiz = []

# Caso o ID não seja encontrado a detecção será preta
cor_classif = (0,0,255)
while ret:
    list_players=[]
    
    #Rodar o mdelo com tracking
    results = model.track(frame, persist = True)

    # Cópia do campo em 2D
    frame_ground = ground.copy()

    for result in results[0]:
        print(result.boxes)
        if result.boxes is not None:
            bbox = result.boxes.xyxy[:4].cpu().numpy()  # Coordenadas da caixa delimitadora (xmin, ymin, xmax, ymax)
            conf = float(result.boxes.conf.cpu().numpy())  # Confiança da detecção
            track_id = result.boxes.id.cpu().numpy()[0] # ID do Tracking
            class_id = result.names[int(result.boxes.cls)]  # ID da classe
            
            if conf > 0.05:  # Considerar apenas detecções com confiança acima de 0.# Desenhar a caixa delimitadora na imagem
                xmin, ymin, xmax, ymax = map(int,(bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3])) # Coordenadas das caixas

                # Escolher os times pelo método de detecção de cor
                #cor_classif = get_team_detect(frame, cor_mandante, cor_visitante, cor_juiz)

                #Escolher os times pelo método de apontamento pelo ID
                cor_classif = get_team_apont(track_id, lista_mandante, lista_visitante, lista_juiz, cor_mandante, cor_visitante, cor_juiz)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), cor_classif, 2)
                cv2.putText(frame, f"ID: {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_classif, 1)

                frame_ground = plot_ground(frame_ground, matrix, xmin, ymin, xmax, ymax, cor_classif)

                list_players.append([xmin,ymin,xmax,ymax,cor_classif])
        else:
            continue

    # Redimensione a imagem "frame" para o tamanho desejado
    new_width = 1720  # Defina a largura desejada
    new_height = 920  # Defina a altura desejada
    frame = cv2.resize(frame, (new_width, new_height))

    # Redimensione a imagem "frame_ground" para o tamanho desejado
    new_width = 360  # Defina a largura desejada
    new_height = 200  # Defina a altura desejada
    frame_ground = cv2.resize(frame_ground, (new_width, new_height))

    # Especifique a posição onde você deseja colocar a imagem "frame_ground" na imagem "frame"
    x_cv, y_cv = 200, 0  # Altere as coordenadas conforme necessário

    # Calcule a região de interesse (ROI) na imagem "frame" onde você deseja colocar a imagem "plane"
    roi = frame[y_cv:y_cv + frame_ground.shape[0], x_cv:x_cv + frame_ground.shape[1]]

    # Mesclar as duas imagens usando cv2.addWeighted()
    alpha = 0.8  # Ajuste o valor alpha para controlar a transparência da imagem "plane"
    beta = 1.0 - alpha
    cv2.addWeighted(frame_ground, alpha, roi, beta, 0, roi)

    cv2.imshow('frame', frame)  
    cv2.waitKey(25)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()