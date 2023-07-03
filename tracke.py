import cv2
from ultralytics import YOLO
from numpy.random import randint

def runTracker(modelName ='yolov8n.pt'):
    # Load a model
    model = YOLO(modelName)  # load an official detection model

    names = model.names
    colors = [[randint(0, 255), randint(0, 255), randint(0, 255)] for _ in range(len(names))]

    video_path = "./videos/corrida19cars-result.mp4"
    save_path = "./resultados/"
    save_path = save_path+video_path.split('/')[-1].replace(".mp4", "-result.mp4")
    cap = cv2.VideoCapture(video_path)
    vid_writer = None

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Track with the model
            results = model.track(source=frame, classes=[0, 1, 2, 3, 4, 6, 7], show=False, tracker="bytetrack.yaml", persist=True) 

            # Draw bounding boxes on the frame
            for boxe in results[0].boxes:
                if boxe.id:
                    box = boxe.xyxy.numpy()
                    x, y, w, h = box[0]
                    cls = int(boxe.cls)
                    id = int(boxe.id)
                    cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), colors[cls], 2)
                    cv2.putText(frame, "id: "+str(id)+" "+str(names[cls]), (int(x), int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[cls], 2)

            # Write the annotated frame to the output video file
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.write(frame)
            else:
                # Create the VideoWriter object
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), isColor=True)
                vid_writer.write(frame)
        else:
            # Break the loop if the end of the video is reached
            break
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
    # Release the video capture object and close the display window
    cap.release()
    #cv2.destroyAllWindows()

runTracker()