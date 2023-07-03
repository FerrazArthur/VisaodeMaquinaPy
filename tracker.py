import cv2
from ultralytics import YOLO
from numpy.random import randint
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return Path(file_path)

def open_directory_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askdirectory()
    return Path(file_path)

def runTracker(modelName='yolov8n.pt', video_path=None, save_dir=None):
    # Carrega o modelo
    model = YOLO(modelName)  # Busca um modelo de deteccao oficial Yolo

    names = model.names
    #cada classe terá uma cor diferente nos bounding boxes
    colors = [[randint(0, 255), randint(0, 255), randint(0, 255)] for _ in range(len(names))]

    # Pede o video a ser processado e o diretorio para salvar o video resultante
    if video_path is None:
        print("Escolha o arquivo que deseja processar:")
        video_path = open_file_dialog()
        print("Video escolhido: ", str(video_path))
        #video_path = input("Enter the path to the video file: ")
    if save_dir is None:
        print("Escolha o diretorio para salvar o video resultante:")
        save_dir = open_directory_dialog()
        print("O video processado será salvo no diretório: ", str(save_dir))
        #save_dir = input("Enter the directory to save the result video: ")
        
    # Realiza uma contagem de quantos arquivos de predicao ja existem no diretorio.
    prediction_files = [f for f in save_dir.glob("prediction*.mp4")]
    prediction_count = len(prediction_files)

    # Gera o nome do arquivo de saida do video resultante
    save_path = save_dir / f"prediction{prediction_count + 1}.mp4"

    # Cria o diretório para salvar o video, caso nao exista
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    vid_writer = None

    #cria o contador de objetos
    objectCounter = {}
    
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    print("Processando video...\n[", end='')
    while cap.isOpened():
        # Lê frame a frame do video de entrada
        success, frame = cap.read()

        #imprime progresso
        count+=1
        if count%(framecount//10) == 0:
            print('.', end='', flush=True)

        if success:
            # realiza o tracking com o modelo bytetrack
            results = model.track(source=frame, classes=[0, 1, 2, 3, 4, 6, 7], show=False, tracker="bytetrack.yaml", persist=True, verbose=False) 

            # Desenha os bounding boxes no frame
            for boxe in results[0].boxes:
                if boxe.id:
                    box = boxe.xyxy.numpy()
                    x, y, w, h = box[0]
                    cls = int(boxe.cls)
                    id = int(boxe.id)
                    cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), colors[cls], 2)
                    cv2.putText(frame, "id: "+str(id)+" "+str(names[cls]), (int(x), int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[cls], 2)
                    #realiza a conta de quantas vezes cada classe aparece
                    
                    if id not in objectCounter.keys():
                        objectCounter[id] = [names[cls]]
                    else:
                        objectCounter[id].append(names[cls])

            # Salva o frame no video resultante
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.write(frame)
            else:
                # Cria objeto vid_writer para salvar o video
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), isColor=True)
                vid_writer.write(frame)
        else:
            # Termina o loop se não houver mais frames
            break
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
    # Libera a captura do video ao fim do processo
    cap.release()
    print("]\nProcessamento finalizado!")
    imprime_resultado(objectCounter)

def find_most_common_element(lst):
    if len(lst) == 0:
        return None
    return max(lst, key=lst.count)

def imprime_resultado(objectCounter):
    classCounter = {}
    for key, value in objectCounter.items():
        #Se for uma lista, encontra a classe mais comum que esse id foi identificado
        if len(value) > 1:
            value = find_most_common_element(value)
        if value in classCounter.keys():
            classCounter[value] += 1
        else:
            classCounter[value] = 1
    
    print("Resultado da contagem de objetos/pessoas no vídeo:")
    for key, value in classCounter.items():
        print(f'{key:10} : {value:5} ocorrências')
runTracker()