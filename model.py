import cv2
import numpy as np
import time

net = cv2.dnn.readNet("model\yolov3.weights", "model\yolov3.cfg")

classes = []
with open("coco.names", "r") as f:      
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))



layers = net.getLayerNames()
output_layer =[layers[layer[0]-1] for layer in net.getUnconnectedOutLayers()]


cap = cv2.VideoCapture(0) 

starting_time = time.time()
img_id = 0
while True:
    (_, img) = cap.read()
    img_id +=1

    img_width = img.shape[1] 
    img_height = img.shape[0]
                                   
    img_blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True) # resmi 4 boyutlu hale getirmek icin.

    net.setInput(img_blob)

    detection_layers = net.forward(output_layer)

    ids_list = []
    boxes_list = []
    confidences_list = []

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            if(confidence > 0.20): # güven aralığı % 80 dan fazla olan nesneleri çiz
                label = classes[predicted_id]
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))


                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

    indexes = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    for max_id in indexes:

        max_class_id = max_id[0]
        box = boxes_list[max_class_id]
        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]
        label = classes[predicted_id]
        confidence = confidences_list[max_class_id]

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each)for each in box_color]

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 2)
        cv2.putText(img, label + " " + str(round(confidence, 2)), (start_x, start_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
    
    
    elapsed_time = time.time() - starting_time
    fps = img_id / elapsed_time
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("Tespit Ekrani", img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()