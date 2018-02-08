# More cascades: openCV's github --> data --> haar cascades
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Greyscale provides more accuracy for face detection.
img = cv2.imread("orp.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img,
scaleFactor=1.5,
minNeighbors=5)

# tells that faces is an n-d numpy array.
print(type(faces))

# will print a numpy array where:
#   first num is where rect starts (x-coord/col)
#   second num is the ycoord of where the face starts
#   third num is the width
#   fourth num is the height
print(faces)

for x, y, w, h in faces:

    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), (3))

resized = cv2.resize(img, (int(img.shape[1]/ 3), int(img.shape[0]/3)))

cv2.imshow("Hey, I've found a face!", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
