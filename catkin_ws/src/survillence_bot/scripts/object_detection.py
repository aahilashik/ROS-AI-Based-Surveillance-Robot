import cv2
import argparse
import numpy as np


def getOutputLayers(net):    
    layerNames = net.getLayerNames()
    return [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPrediction(image, classID, confidence, x, y, w, h):
    label = str(classes[classID])
    color = COLORS[classID]

    image = cv2.rectangle(image, (x,y), (x + w, y + h), color, 2)
    image = cv2.putText(image, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

confThreshold = 0.5
nmsThreshold  = 0.4

classesPath = "/home/ubuntussd/catkin_ws/src/survillence_bot/config/compVis/classes.txt"
weightsPath = "/home/ubuntussd/catkin_ws/src/survillence_bot/config/compVis/yolov3.weights"
configPath  = "/home/ubuntussd/catkin_ws/src/survillence_bot/config/compVis/yolov3.cfg"


with open(classesPath, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

reqClasses = classes # ["bicycle", "dog", "truck"]

net = cv2.dnn.readNet(weightsPath, configPath)

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def detectObjects(image, targetClass=reqClasses, show=False):
    height, width, depth = image.shape
    
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputLayers(net))
    
    classIDs, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confThreshold and classes[classID] in reqClasses:
                centerX, centerY = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(centerX - w / 2), int(centerY - h / 2)
                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    data = {}
    for i in indices.reshape(-1):
        x, y, w, h = boxes[i]
	data[str(classes[classIDs[i]])] = [x, y, w, h]
        image = drawPrediction(image, classIDs[i], confidences[i], x, y, w, h)

    if show: 
        cv2.imshow("object detection", image)
        cv2.waitKey()
    else: return data, image


if __name__ == '__main__': 
    image = cv2.imread("/home/ubuntussd/catkin_ws/src/survillence_bot/scripts/dog.jpg")
    detectObjects(image, show=True)
