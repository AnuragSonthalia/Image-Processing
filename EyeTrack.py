import cv2

class EyeTracker:
    def __init__(self, faceCascadePath,eyeCascadePath):
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)
        self.eyeCascade = cv2.CascadeClassifier(eyeCascadePath)

    def track(self, image):
        faceRects = self.faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
        rects=[]
        for (fx,fy,fw,fh) in faceRects:
            faceROI=image[fy:fy+fh,fx:fx+fw]
            rects.append((fx,fy,fx+fw,fy+fh))

            eyeRects= self.eyeCascade.detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=10,minSize=(20,20), flags=cv2.CASCADE_SCALE_IMAGE)

            for (ex,ey,ew,eh) in eyeRects:
                rects.append((fx+ex, fy+ey, fx + ex + fw, fy + ey + fh))

        return rects
