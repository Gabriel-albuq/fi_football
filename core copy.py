import os
import cv2
from ultralytics import YOLO
import numpy as np
import calib
from collections import Counter

class ClassColorInertia:
    def __init__(self, track_id, inertia, color):
        self.list_color = [color]
        self.inertia = inertia
        self.track_id = track_id

    def add_element(self, color):
        if len(self.list_color) == self.inertia:
            self.list_color.pop() # Se a lista atingiu a capacidade máxima, remove o último elemento

        self.list_color.insert(0, color) # Adiciona o novo elemento na primeira posição
    
    def get_list(self):
        return self.list_color
    
    def get_inertia(self):
        return self.inertia
    
    def get_track_id(self):
        return self.track_id
    
    def get_inertia_color(self):
        if not self.list_color:
            return [0,0,0]

        counter = Counter(tuple(color) for color in self.list_color)
        most_common_element = counter.most_common(1)[0][0]
        return most_common_element
    

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    inc = 10

    h = hsvC[0][0][0]
    if h + inc >= 255:
        h_l = h - inc
        h_u = 255
    elif h - inc < 1:
        h_l = 1
        h_u = h + inc
    else: 
        h_l = h - inc
        h_u = h + inc

    s = hsvC[0][0][1] 
    if s + inc >= 255:
        s_l = s - inc
        s_u = 255
    elif s - inc < 1:
        s_l = 1
        s_u = s + inc
    else: 
        s_l = s - inc
        s_u = s + inc

    v = hsvC[0][0][2] 
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

    return lowerLimit, upperLimit

def get_team_detect(frame, track_id, cor_mandante, cor_visitante, cor_juiz, xmin, ymin, xmax, ymax):
    frame_crop = frame[ymin:ymax, xmin:xmax]

    # Ajustar para pegar só a camisa pela proporção do corpo
    inferior_pe = ymax
    topo_cabeca = ymin
    
    div_corpo = int((topo_cabeca - inferior_pe) / 8.5)
    ymax_camisa = inferior_pe + 5 * div_corpo
    ymin_camisa = topo_cabeca - 1 * div_corpo

    hsvImage = cv2.cvtColor(frame[ymin_camisa:ymax_camisa, xmin:xmax], cv2.COLOR_BGR2HSV)
    #cv2.imshow("Cor camisa",hsvImage)
    #cv2.waitKey(2500)
    
    # Mandante
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

def get_color_inertia(List_ClassColorInertia, track_id, cor_classif):
    inertia = 20
    # Encontrar o objeto com o track_id_to_check
    found_object = next((obj for obj in List_ClassColorInertia if obj.get_track_id() == track_id), None)
    if found_object is not None:
        found_object.add_element(cor_classif)
    else:
        List_ClassColorInertia.append(ClassColorInertia(track_id, inertia, cor_classif))

    found_object = next((obj for obj in List_ClassColorInertia if obj.get_track_id() == track_id), None)
    color_inertia = found_object.get_inertia_color()

    return color_inertia

# Caminho do vídeo
video_path = os.path.join('.', 'data', 'video.avi')
ground_path = 'data/dst.jpg'

# Vídeo de exportação
output_video_path = os.path.join('.', 'data', 'video_export.mp4')
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'H264' or 'mp4v' for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, 25, (1720, 920))

# Definir se vai pltoar os pontos ou passar a matriz diretamente
switch_matrix = 1

if switch_matrix == 0:
    matrix = calib.create_points(video_path, ground_path)
    print(matrix)
else: matrix = np.array([[    0.31894,     0.66294,       169.5],
                        [   0.020267,      1.5178,     -123.68],
                        [ 4.4284e-05,    0.001466,           1]])

# Campo em 2D
ground = cv2.imread(ground_path, -1)

#Vídeo
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Modelo de detecção
model = YOLO("best.pt")

#Cores dos times, goleiro e juiz
cor_mandante = [111, 47, 38]
cor_visitante = [255, 251, 255]
cor_juiz = [19, 48, 191]

# Para o método de times por apontamento, pegar os IDs e escrever aqui
lista_mandante = [2,3,11,12,20]
lista_visitante = [1,4,5,6,9,7,10,14]
lista_juiz = []

# Caso o ID não seja encontrado a detecção será preta
List_ClassColorInertia = []
cor_classif = (0,0,255)
while ret:
    list_players=[]
    
    #Rodar o mdelo com tracking
    results = model.track(frame, persist = True)

    # Cópia do campo em 2D
    frame_ground = ground.copy()

    for result in results[0]:
        if result.boxes is not None:
            bbox = result.boxes.xyxy[:4].cpu().numpy()  # Coordenadas da caixa delimitadora (xmin, ymin, xmax, ymax)
            conf = float(result.boxes.conf.cpu().numpy())  # Confiança da detecção
            track_id = result.boxes.id.cpu().numpy()[0] # ID do Tracking
            class_id = result.names[int(result.boxes.cls)]  # ID da classe
            
            if conf > 0.7:  # Considerar apenas detecções com confiança acima de 0.# Desenhar a caixa delimitadora na imagem
                xmin, ymin, xmax, ymax = map(int,(bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3])) # Coordenadas das caixas

                # Escolher os times pelo método de detecção de cor
                cor_classif = get_team_detect(frame, track_id, cor_mandante, cor_visitante, cor_juiz, xmin, ymin, xmax, ymax)

                #Escolher os times pelo método de apontamento pelo ID
                #cor_classif = get_team_apont(track_id, lista_mandante, lista_visitante, lista_juiz, cor_mandante, cor_visitante, cor_juiz)

                cor_classif = get_color_inertia(List_ClassColorInertia, track_id, cor_classif)

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

    out.write(frame)
    cv2.imshow('frame', frame)  
    cv2.waitKey(25)

    ret, frame = cap.read()

out.release()
cap.release()
cv2.destroyAllWindows()