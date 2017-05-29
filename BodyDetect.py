import cv2

class BodyDetector:
    def __init__(self,bodyCascadePath):
        self.bodyCascade=cv2.CascadeClassifier(bodyCascadePath)

    def detect(self,image, scalefactor=1.05, minneighbors=5, minsize=(30, 30)):
        rects=self.bodyCascade.detectMultiScale(image,scaleFactor=scalefactor,minNeighbors=minneighbors,minSize=minsize,flags=cv2.CASCADE_SCALE_IMAGE)
        return rects