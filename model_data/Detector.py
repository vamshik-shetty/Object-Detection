import cv2
import numpy as np
import time

np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath, logPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        self.logPath = logPath

        #############################

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()
        self.detected_objects = []
        self.overall_detected_objects = set()

    def readClasses(self):
        with open(self.classesPath, "r") as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, ' __Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))


    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
   
        if not cap.isOpened():
            print("Error opening file...")
            return

        width = int(cap.get(3))
        height = int(cap.get(4))
        new_width = 1300
        new_height = int((new_width / width) * height)

        while True:
            (success, image) = cap.read()

            if not success:
                break

            image = cv2.resize(image, (new_width, new_height))
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdx) != 0:
                frame_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                frame_info = f"Time: {frame_time}, Detected Objects: "
                frame_objects = set()
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                    #############################

                    lineWidth = min(int(w * 0.3), int(h * 0.3))

                    cv2.line(image, (x,y), (x + lineWidth, y), classColor, thickness= 5)
                    cv2.line(image, (x,y), (x, y + lineWidth), classColor, thickness= 5)

                    cv2.line(image, (x + w,y), (x + w - lineWidth, y), classColor, thickness= 5)
                    cv2.line(image, (x + w,y), (x + w, y + lineWidth), classColor, thickness= 5)

                    ###############################

                    cv2.line(image, (x,y + h), (x + lineWidth, y + h), classColor, thickness= 5)
                    cv2.line(image, (x,y + h), (x, y + h - lineWidth), classColor, thickness= 5)

                    cv2.line(image, (x + w,y + h), (x + w - lineWidth, y + h), classColor, thickness= 5)
                    cv2.line(image, (x + w,y + h), (x + w, y + h - lineWidth), classColor, thickness= 5)

                    frame_objects.add(classLabel)
                    self.overall_detected_objects.add(classLabel)

                frame_info += ", ".join(frame_objects)
                self.detected_objects.append(frame_info)

            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()

        with open(self.logPath, 'w') as log_file:
            log_file.write("\nObject Detection Using Deep Learning:\n")
            for obj_info in self.detected_objects:
                log_file.write(f"• {obj_info}\n")

            log_file.write("\nOverall Detected Objects:\n")
            for obj in self.overall_detected_objects:
                log_file.write(f"• {obj}\n")    

        with open(self.logPath, 'r') as log_file:
            log_content = log_file.read()
            print(log_content)

        cv2.destroyAllWindows()