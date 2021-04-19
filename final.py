import cv2
import numpy as np
import glob
from skimage import measure
import matplotlib.pyplot as plt
import os

net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')
images_path = glob.glob(r"pics\*.jpg")

images_path = [i for i in images_path if i.endswith('.jpg')]
images_path = [i.split('\\')[1] for i in images_path] 
images_path = [i.split('.')[0] for i in images_path] 
images_path.sort(key=int)
images_path = ["pics\\"+i+".jpg"for i in images_path] 

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()


font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

survivor_detected = []
survivor_count = 0
casualty_detect = False
casualty_count= 0

final_count = []

for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img,( 416, 416))
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                if class_id == 1:
                    survivor_detected.append(img_path)
                if class_id == 0:
                    survivor_detected.append(img_path)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if class_ids[i] == 1:
                color = [50 , 255, 50]

            else:
                color = [50 , 50, 255]
                casualty_detect = True
                
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
    cv2.imshow('Image', img)
    
    k = cv2.waitKey(2)
    if k==ord('q'):    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue

cv2.destroyAllWindows()

survivor_detected = list(dict.fromkeys(survivor_detected))

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = measure.compare_ssim(imageA, imageB,multichannel=True)
    # setup the figure
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    if s < 0.78:
        return True
    

n = len(survivor_detected)-1

final_count.append("pics\\0.jpg")

for x in range(1, n):
    imageA = cv2.imread(survivor_detected[x-1])
    imageB = cv2.imread(survivor_detected[x])
    print("comparing "+str(survivor_detected[x-1])+" and "+str(survivor_detected[x]))
    if(compare_images(imageA, imageB)):
        final_count.append("pics\\"+str(x)+".jpg")

        
print("done comapring")


for img_path in final_count:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img,( 416, 416))
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if class_ids[i] == 1:
                color = [50 , 255, 50]
                survivor_count+=1

            else:
                color = [50 , 50, 255]
                casualty_count+=1
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
    cv2.putText(img, "Survivor Count: " + str(survivor_count), (0, 25), 0, 0.8, (0, 255, 0), 2)
    if casualty_detect == 0:
        cv2.putText(img, "Casualty: Not detected" , (0, 55), 0, 0.8, (0, 255, 0), 2) 
    else:
        cv2.putText(img, "Casualty: Detected" , (0, 55), 0, 0.8, (0, 0, 255), 2)
    cv2.putText(img, "Casualty Count: " + str(casualty_count), (0, 85), 0, 0.8, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    
    k = cv2.waitKey(0)
    if k==ord('q'):    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue

if(casualty_detect):
    print("There were casualty detected")
else:
    print("No casualty detected")
    
print("There were "+str(survivor_count)+" Survivors")        

cv2.destroyAllWindows()