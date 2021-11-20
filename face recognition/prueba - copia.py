import face_recognition
import os
import cv2
import sys
import numpy as np
import glob

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'knowImages/')
#know_Face_dir = ""
known_face_encodings = []
known_face_names = []

#make an array of all the saved jpg files' paths
list_of_files = [f for f in glob.glob(path+'*.jpg')]
#find number of known faces
number_files = len(list_of_files)
#We copy list of the names of all photos of known faces
names = list_of_files.copy()

for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    known_face_encodings.append(globals()['image_encoding_{}'.format(i)])
    # Create array of known names
    names[i] = names[i].replace("knowImages\\", "")
    names[i] = names[i].replace(".jpg", "") 
    known_face_names.append(names[i])

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Know Images (esto puedo borrarlo si funciona lo otro)
imagePath = "knowImages\carlos.jpg"
carlosImage = face_recognition.load_image_file(imagePath)
carlosImage = cv2.cvtColor(carlosImage, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(carlosImage)[0]
encoderFaceCar = face_recognition.face_encodings(carlosImage)[0]
cv2.rectangle(carlosImage, (faceLoc[3],faceLoc[0]),(faceLoc[1], faceLoc[2]),(0,255,0),2)

print(known_face_names)

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
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]


    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),2)
        try:
            cv2.rectangle(frame, (x,y+h-35),(x+w, y+h),(0,255,0),cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y+h - 6), font, 1.0, (255, 255, 255), 1)
        except:
            continue

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