# this file not related with project.     "only to check if camera is working or not"
import cv2

cap = cv2.VideoCapture()
if not cap.isOpened():
    print("Camera failed to open")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Test", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
