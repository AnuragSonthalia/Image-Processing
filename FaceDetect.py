import cv2

class FaceDetector:
    def __init__(self, faceCascadePath):
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    def detect(self, image, scalefactor=1.2, minneighbors=5, minsize=(30, 30)):
        rects = self.faceCascade.detectMultiScale(image, scaleFactor=scalefactor, minNeighbors=minneighbors,minSize=minsize, flags = cv2.CASCADE_SCALE_IMAGE)
        return rects
