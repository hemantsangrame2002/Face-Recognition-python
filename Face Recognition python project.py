import cv2

# Load Haar cascade for face detection
face_cap = cv2.CascadeClassifier(
    "C:/Users/hemant/AppData/Roaming/Python/Python312/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

# Open the webcam (0 is usually the default webcam)
video_cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not video_cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        red, video_data = video_cap.read()
        
        if not red:
            print("Error: Could not read frame.")
            break

        # Convert frame to grayscale
        col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cap.detectMultiScale(col, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(video_data, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("video_live", video_data)

        # If the user presses 'a', break the loop
        if cv2.waitKey(100) == ord("a"):
            break

# Release the webcam and close all OpenCV windows
video_cap.release()
cv2.destroyAllWindows()
