import numpy as np
import cv2
import os

drawing = False # true if mouse is pressed
red_circle = False
tela = "-"
src_x, src_y = -1,-1
dst_x, dst_y = -1,-1

src_list = []
dst_list = []

# mouse callback function
def select_points_src(event,x,y,flags,param):
    global src_x, src_y, drawing, tela, red_circle
    if event == cv2.EVENT_MOUSEMOVE:
        src_x, src_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        red_circle = True
        tela = "src_copy"
        src_x, src_y = x,y
        #cv2.circle(src_copy,(x,y),5,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# mouse callback function
def select_points_dst(event,x,y,flags,param):
    global dst_x, dst_y, drawing, tela, red_circle
    if event == cv2.EVENT_MOUSEMOVE:
        dst_x, dst_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        red_circle = True
        tela = "dst_copy"
        dst_x, dst_y = x,y
        #cv2.circle(dst_copy,(x,y),5,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def create_transformation(src_list, dst_list):
    np_src_list = np.array(src_list, dtype=np.float32)
    np_dst_list = np.array(dst_list, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np_src_list, np_dst_list)
    print(matrix)
    return matrix

def create_points(video_path, dst_path):

    #video_path = os.path.join('.', 'data', 'test2.mp4')
    cap = cv2.VideoCapture(video_path)
    ret, src = cap.read()

    src_copy = src.copy()
    cv2.namedWindow('src')
    cv2.moveWindow("src", 80,80)
    cv2.setMouseCallback('src', select_points_src)

    #dst_path = 'data/dst.jpg'
    dst = cv2.imread(dst_path, -1)
    dst_copy = dst.copy()
    cv2.namedWindow('dst')
    cv2.moveWindow("dst", 780,80)
    cv2.setMouseCallback('dst', select_points_dst)

    result = 0
    while(1):
        # Abrir o vídeo
        cv2.imshow('src',src_copy)

        #Abrir o campo
        cv2.imshow('dst',dst_copy)

        # Coordenadas em tempo real
        cv2.rectangle(src_copy, (0, 0), (200, 30), (255, 255, 255), -1)
        cv2.rectangle(dst_copy, (0, 0), (200, 30), (255, 255, 255), -1)
        cv2.putText(src_copy, f'Coord: ({src_x}, {src_y})', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(dst_copy, f'Coord: ({dst_x}, {dst_y})', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        k = cv2.waitKey(1) & 0xFF
        
        if red_circle == True:
            if k == ord('s'):
                if tela == "src_copy":
                    ponto = src_x,src_y
                    src_list.append([src_x,src_y])
                    cv2.circle(src_copy,ponto,5,(0,255,0),-1)
                else: 
                    ponto = dst_x,dst_y
                    dst_list.append([dst_x,dst_y])
                    cv2.circle(dst_copy,ponto,5,(0,255,0),-1)

                print('save points')
                print(f"src points: {src_list}")
                print(f"dst points: {dst_list}")
                
        if k == ord('d'):
            matrix = create_transformation(src_list, dst_list)
            result = 1
            k = cv2.waitKey(1) & 0xFF
        
        if k == ord('t'):
            cv2.circle(src_copy,(src_x,src_y),5,(255,0,0),-1)
            original_point = np.float32([[src_x,src_y, 1]])
            print(f"Original Point: {original_point}")

            transform_point = np.dot(matrix, original_point.T).T
            transform_point = transform_point / transform_point[0, 2]

            print(f"Transform Point: {transform_point}")

            cv2.circle(dst_copy,(int(transform_point[0][0]), int(transform_point[0][1])),5,(0,255,0),-1)

        if k == ord('e'):
            break
        
    cv2.destroyAllWindows()

    return(matrix)
