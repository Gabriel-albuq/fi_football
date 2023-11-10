import os
import cv2
from ultralytics import YOLO
import numpy as np
import calib

def display_color_squares(name, colors, square_size=(100, 100)):
    # Create an empty image to hold all color squares
    result_image = np.zeros((square_size[0], len(colors) * square_size[1], 3), dtype=np.uint8)

    # Iterate through colors and create a square for each color
    for i, color in enumerate(colors):
        color_square = np.zeros((square_size[0], square_size[1], 3), dtype=np.uint8)
        color_square[:, :] = color
        result_image[:, i * square_size[1] : (i + 1) * square_size[1]] = color_square

    # Display the result image
    cv2.imshow(name, result_image)

def get_limits(name, color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    h = hsvC[0][0][0]  # Get the hue value
    s = hsvC[0][0][1] 
    v = hsvC[0][0][2] 
    inc = 20

    if h + inc >= 255:
        h_l = h - inc
        h_u = 255
    elif h - inc < 1:
        h_l = 1
        h_u = h + inc
    else: 
        h_l = h - inc
        h_u = h + inc

    if s + inc >= 255:
        s_l = s - inc
        s_u = 255
    elif s - inc < 1:
        s_l = 1
        s_u = s + inc
    else: 
        s_l = s - inc
        s_u = s + inc

    if v + inc >= 255:
        v_l = v - inc
        v_u = 255
    elif v - inc < 1:
        v_l = 1
        v_u = v + inc
    else: 
        v_l = v - inc
        v_u = v + inc

    lowerLimit = np.array([h_l, s_l, v_l], dtype=np.uint8)
    upperLimit = np.array([h_u, s_u, v_u], dtype=np.uint8)

    colors =  lowerLimit, hsvC, upperLimit
    display_color_squares(name, colors)

    return lowerLimit, upperLimit

def get_team_detect(frame, cor_mandante, cor_visitante, cor_juiz):
    frame_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cor_mandante_hsv = cv2.cvtColor(np.uint8([[cor_mandante]]), cv2.COLOR_BGR2HSV)[0][0]
    cor_visitante_hsv = cv2.cvtColor(np.uint8([[cor_visitante]]), cv2.COLOR_BGR2HSV)[0][0]
    cor_juiz_hsv = cv2.cvtColor(np.uint8([[cor_juiz]]), cv2.COLOR_BGR2HSV)[0][0]

    #Mandante
    name = "Mandante"
    lowerLimit, upperLimit = get_limits(name, color=cor_mandante)
    mask = cv2.inRange(frame_crop, lowerLimit, upperLimit)
    count_mandante = cv2.countNonZero(mask)
    print(f'{name}: {count_mandante} - {lowerLimit,cor_mandante_hsv, upperLimit}')

    #Visitante
    name = "Visitante"
    lowerLimit, upperLimit = get_limits(name, color=cor_visitante)
    mask = cv2.inRange(frame_crop, lowerLimit, upperLimit)
    count_visitante = cv2.countNonZero(mask)
    print(f'{name}: {count_visitante} - {lowerLimit, cor_visitante_hsv, upperLimit}')

    #Juiz
    name = "Juiz"
    lowerLimit, upperLimit = get_limits(name, color=cor_juiz)
    mask = cv2.inRange(frame_crop, lowerLimit, upperLimit)
    count_juiz = cv2.countNonZero(mask)
    print(f'{name}: {count_juiz} - {lowerLimit, cor_juiz_hsv, upperLimit}')

    lista_cor = [cor_mandante, cor_visitante, cor_juiz]
    lista_count = [count_mandante, count_visitante, count_juiz]

    posicao_lista = lista_count.index(max(lista_count))
    cor_classif = lista_cor[posicao_lista] 

    return posicao_lista, frame_crop, cor_mandante_hsv, cor_visitante_hsv, cor_juiz_hsv


frame_path = 'data/juiz.png'
frame = cv2.imread(frame_path, -1)

# Obtém as dimensões da imagem
xmin = 0
ymin = 0
xmax = frame.shape[1]
ymax = frame.shape[0]

# Ajustar para pegar só a camisa pela proporção do corpo
inferior_pe = ymax
topo_cabeca = ymin

div_corpo = int((topo_cabeca - inferior_pe) / 8.5)
ymax = inferior_pe + 5 * div_corpo
ymin = topo_cabeca - 1 * div_corpo

# Extrai a região de interesse (ROI) da imagem
roi = frame.copy()
roi = roi[ymin:ymax, xmin:xmax]

#Cores dos times, goleiro e juiz
cor_mandante = [111, 47, 38]
cor_visitante = [255, 251, 255]
cor_juiz = [19, 48, 191]

cor_classif, frame_crop, cor_mandante_hsv, cor_visitante_hsv, cor_juiz_hsv = get_team_detect(frame, cor_mandante, cor_visitante, cor_juiz)
print(cor_classif)
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

cv2.imshow('frame_crop', roi)

cv2.waitKey(25000)

cv2.destroyAllWindows()