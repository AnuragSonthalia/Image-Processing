from BodyDetect import BodyDetector
import cv2
import argparse
import imutils1

ap=argparse.ArgumentParser()
ap.add_argument("-f","--body",required=True,help="path to where the body cascade resides")
ap.add_argument("-i","--image",required=True,help="path to where the image file resides")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
image=imutils1.resize(image,width=600)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
bd=BodyDetector(args["body"])
bodyRects=bd.detect(gray,scalefactor=1.01,minneighbors=1,minsize=(3,3))

for (x,y,w,h) in bodyRects:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("Body",image)
cv2.waitKey(0)