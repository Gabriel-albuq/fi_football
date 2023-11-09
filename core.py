import os
import cv2
from ultralytics import YOLO
import numpy as np

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

def plane(players,ball=None):
    coptemp=ground.copy()
    matrix=np.array([[-3.43625050e-01, -3.50826328e+00,  5.63367146e+02],
                    [ 8.77547187e-02, -6.25669566e+00,  7.70011960e+02],
                    [ 1.40226685e-04, -1.14567177e-02,  1.00000000e+00]])
    
    for p in players:
        x_p = (p[0] + p[2]) / 2
        y_p = p[3]
        pts3 = np.float32([[x_p,y_p, 1]])
        #cv2.circle(frame,(int(x_p),int(y_p)), 5, (255,0,0),-1)

        pts3o = np.dot(matrix, pts3 .T).T
        pts3o = pts3o / pts3o[0, 2]
        
        xmin1=int(pts3o[0][0])
        ymin1=int(pts3o[0][1])

        #pts3o=cv2.perspectiveTransform(pts3[None, :, :],matrix)
        #xmin1=int(pts3o[0][0][0])
        #ymin1=int(pts3o[0][0][1])
        pp=(xmin1,ymin1)
        cv2.circle(coptemp,pp, 10, p[4],-1)

            #print hakm
            #cv2.circle(coptemp,pp, 15, (0,0,255),-1)
    #if len(ball) !=0:
        
    #    xb=ball[0]+int(ball[2]/2)
    #    yb=ball[1]+int(ball[3]/2)
    #    pts3ball = np.float32([[xb,yb]])
    #    pts3b=cv2.perspectiveTransform(pts3ball[None, :, :],matrix)
    #    x2=int(pts3b[0][0][0])
    #    y2=int(pts3b[0][0][1])
    #    pb=(x2,y2)
    #    cv2.circle(coptemp,pb, 15, (0,0,0),-1)
    return coptemp

ground = cv2.imread('data/dst.jpg', -1)

video_path = os.path.join('.', 'data', 'test2.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

model = YOLO("yolov8n.pt")

cor_mandante = [40, 56, 244]
cor_visitante = [0, 192, 255]
cor_juiz = [0, 0, 0]

lista_mandante = [2,3,11,12,20]
lista_visitante = [1,4,5,6,9,7,10,14]
lista_juiz = []

cor_classif = (0,0,255)
while ret:
    players=[]
    results = model.track(frame, persist = True)

    for result in results[0]:
        bbox = result.boxes.xyxy[:4].cpu().numpy()  # Coordenadas da caixa delimitadora (xmin, ymin, xmax, ymax)
        conf = float(result.boxes.conf.cpu().numpy())   # Confiança da detecção
        track_id = result.boxes.id.cpu().numpy()[0]
        class_id = result.names[int(result.boxes.cls)]  # ID da classe
        
        if conf > 0.1:  # Considerar apenas detecções com confiança acima de 0.# Desenhar a caixa delimitadora na imagem
            xmin, ymin, xmax, ymax = map(int,(bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]))  

            #media_cores = cv2.mean(frame[int(ymin):int(ymax), int(xmin):int(xmax)])
            #print("Média de cores (BGR):", media_cores[:3])

            # Escolher os times pelo método de detecção de cor
            #cor_classif = get_team_detect(frame, cor_mandante, cor_visitante, cor_juiz)

            #Escolher os times pelo método de apontamento pelo ID
            cor_classif = get_team_apont(track_id, lista_mandante, lista_visitante, lista_juiz, cor_mandante, cor_visitante, cor_juiz)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), cor_classif, 2)
            cv2.putText(frame, f"Track ID: {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_classif, 1)

            players.append([xmin,ymin,xmax,ymax,cor_classif])

    p=plane(players)

    # Redimensione a imagem "p" para o tamanho desejado
    new_width = 1720  # Defina a largura desejada
    new_height = 920  # Defina a altura desejada
    frame = cv2.resize(frame, (new_width, new_height))

    # Redimensione a imagem "p" para o tamanho desejado
    new_width = 360  # Defina a largura desejada
    new_height = 200  # Defina a altura desejada
    p = cv2.resize(p, (new_width, new_height))

    # Especifique a posição onde você deseja colocar a imagem "plane" na imagem "frame"
    x_cv, y_cv = 200, 0  # Altere as coordenadas conforme necessário

    # Calcule a região de interesse (ROI) na imagem "frame" onde você deseja colocar a imagem "plane"
    roi = frame[y_cv:y_cv+p.shape[0], x_cv:x_cv+p.shape[1]]

    # Mesclar as duas imagens usando cv2.addWeighted()
    alpha = 0.8  # Ajuste o valor alpha para controlar a transparência da imagem "plane"
    beta = 1.0 - alpha
    cv2.addWeighted(p, alpha, roi, beta, 0, roi)

    #cv2.imshow('plane',p)
    cv2.imshow('frame', frame)  
    cv2.waitKey(25)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()