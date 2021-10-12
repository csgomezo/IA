import face_recognition
import os
import cv2
import sys
import numpy as np
import glob


path = os.path.join(dirname, 'knowImages/')
#know_Face_dir = ""
known_face_encodings = []
known_face_names = []

#make an array of all the saved jpg files' paths
list_of_files = [f for f in glob.glob(path+'*.jpg')]
#find number of known faces
number_files = len(list_of_files)

names = list_of_files.copy()

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Know Images
imagePath = "knowImages\pp.jpg"
carlosImage = face_recognition.load_image_file(imagePath)
carlosImage = cv2.cvtColor(carlosImage, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(carlosImage)[0]
encoderFaceCar = face_recognition.face_encodings(carlosImage)[0]
cv2.rectangle(carlosImage, (faceLoc[3],faceLoc[0]),(faceLoc[1], faceLoc[2]),(0,255,0),2)


while True:
    #capture frame-by-frame
    ret,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

     # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
    try:
        encoderFace = face_recognition.face_encodings(frame)[0]    
        results = face_recognition.compare_faces([encoderFaceCar],encoderFace)
    except:
        continue

    print(results)
    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


'''

imagePath = "knowImages\pp.jpg"
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Foto mía",gray)
#print(gray)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

a = len(faces)
print(f"Found {a} faces!")

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
cv2.waitKey(0)

'''