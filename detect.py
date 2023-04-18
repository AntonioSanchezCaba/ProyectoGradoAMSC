from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("C:/Users/anton/ultralytics/runs/detect/train3/weights/best.pt")
model.predict(source="0", show=True, conf=0.5, classes=[0, 1])
#___________________________________________________________________________________________________

#___________________________________________________________________________________________________
# import cv2
# from ultralytics import YOLO
# import random
# import os

# model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'best.pt')

# # Load a model
# model = YOLO(model_path)  # load a custom mode

# threshold = 0.5

# class_name_dict = {0: 'Lion', 1: 'Tiger'}

# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(2)]
# cap = cv2.VideoCapture(0)

# while True:
#     _,frame =cap.read()

#     results = model(frame)
#     for result in results:
#         detections = []
#         for r in result.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             x1 = int(x1)
#             x2 = int(x2)
#             y1 = int(y1)
#             y2 = int(y2)
#             class_id = int(class_id)
#             detections.append([x1, y1, x2, y2, score])

#         for detection in detections:
#             x1, y1, x2, y2, score = detection
#             if score > threshold:
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
#                 cv2.putText(frame,  class_name_dict[int(class_id)].upper(), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     cv2.imshow('img', frame)
#     if cv2.waitKey(10) & 0xFF == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()