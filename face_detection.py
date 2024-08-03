import cv2

# Load the pre-trained face detection model
f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the default camera (usually the webcam)
capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    _, img = capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = f_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (300, 0, 0), 1)

    # Display the resulting frame
    cv2.imshow('Face Detection', img)

    # Exit if the 'ESC' key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the capture and close any OpenCV windows
capture.release()
cv2.destroyAllWindows()
