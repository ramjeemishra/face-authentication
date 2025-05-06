# import cv2
# import os

# def AuthenticateFace():
#     flag = 0

#     if not os.path.exists('auth\\trainer\\trainer.yml'):
#         print("Trained model not found.")
#         return 0

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read('auth\\trainer\\trainer.yml')

#     cascadePath = "auth\\haarcascade_frontalface_default.xml"
#     faceCascade = cv2.CascadeClassifier(cascadePath)
#     if faceCascade.empty():
#         print("Failed to load Haar cascade.")
#         return 0

#     cam = cv2.VideoCapture(0)
#     if not cam.isOpened():
#         print("Could not open camera.")
#         return 0

#     cam.set(3, 640)
#     cam.set(4, 480)

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     names = ['', '', 'Ramjee']
#     minW = 0.1 * cam.get(3)
#     minH = 0.1 * cam.get(4)

#     while True:
#         ret, img = cam.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(
#             gray,
#             scaleFactor=1.2,
#             minNeighbors=5,
#             minSize=(int(minW), int(minH)),
#         )

#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             id, accuracy = recognizer.predict(gray[y:y+h, x:x+w])
#             if accuracy < 100:
#                 id_text = names[id]
#                 acc_text = f"  {round(100 - accuracy)}%"
#                 flag = 1
#             else:
#                 id_text = "unknown"
#                 acc_text = f"  {round(100 - accuracy)}%"
#                 flag = 0

#             cv2.putText(img, str(id_text), (x+5, y-5), font, 1, (255, 255, 255), 2)
#             cv2.putText(img, acc_text, (x+5, y+h-5), font, 1, (255, 255, 0), 1)

#         cv2.imshow('camera', img)

#         k = cv2.waitKey(10) & 0xff
#         if k == 27 or flag == 1:
#             break

#     cam.release()
#     cv2.destroyAllWindows()
#     return flag



# import cv2
# import os

# def AuthenticateFace():
#     flag = 0

#     # Load face recognizer
#     if not os.path.exists('auth\\trainer\\trainer.yml'):
#         print("Trained model not found.")
#         return 0

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read('auth\\trainer\\trainer.yml')

#     cascadePath = "auth\\haarcascade_frontalface_default.xml"
#     faceCascade = cv2.CascadeClassifier(cascadePath)
#     if faceCascade.empty():
#         print("Failed to load Haar cascade.")
#         return 0

#     # Load intro video
#     video = cv2.VideoCapture("facerecognition.mp4")
#     if not video.isOpened():
#         print("Failed to open video.")
#         return 0

#     # Open webcam
#     cam = cv2.VideoCapture(0)
#     if not cam.isOpened():
#         print("Could not open camera.")
#         return 0

#     cam.set(3, 320)
#     cam.set(4, 240)

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     names = ['', '', 'Ramjee']
#     minW = 0.1 * cam.get(3)
#     minH = 0.1 * cam.get(4)

#     while True:
#         # Read from video
#         ret_vid, frame_vid = video.read()
#         if not ret_vid:
#             video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
#             ret_vid, frame_vid = video.read()

#         frame_vid = cv2.resize(frame_vid, (320, 240))

#         # Read from webcam
#         ret_cam, img = cam.read()
#         if not ret_cam:
#             print("Failed to grab frame.")
#             break

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))

#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             id, accuracy = recognizer.predict(gray[y:y+h, x:x+w])
#             confidence = 100 - accuracy  # Higher = better match

#             if confidence > 20:
#                 id_text = names[id] if id < len(names) else "Unknown"
#                 acc_text = f"{round(confidence)}%"
#                 flag = 1
#                 cv2.putText(img, "Face Matched!", (x, y - 10), font, 0.6, (0, 255, 0), 2)
#             else:
#                 id_text = "Unknown"
#                 acc_text = f"{round(confidence)}%"
#                 flag = 0

#             cv2.putText(img, id_text, (x+5, y-5), font, 0.6, (255, 255, 255), 2)
#             cv2.putText(img, acc_text, (x+5, y+h-5), font, 0.6, (255, 255, 0), 1)

#         img = cv2.resize(img, (320, 240))

#         # Combine video frame and cam frame horizontally
#         combined = cv2.hconcat([frame_vid, img])

#         cv2.imshow("Authentication", combined)

#         k = cv2.waitKey(30) & 0xff
#         if k == 27 or flag == 1:
#             break

#     cam.release()
#     video.release()
#     cv2.destroyAllWindows()
#     return flag

# AuthenticateFace()


import numpy as np
import cv2
import os

def AuthenticateFace():
    flag = 0

    # Load recognizer
    if not os.path.exists('auth\\trainer\\trainer.yml'):
        print("Trained model not found.")
        return 0

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('auth\\trainer\\trainer.yml')

    # Load Haar cascade
    cascadePath = "auth\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    if faceCascade.empty():
        print("Failed to load Haar cascade.")
        return 0

    # Open video
    video = cv2.VideoCapture("facerecognition.mp4")
    if not video.isOpened():
        print("Failed to open intro video.")
        return 0

    # Open webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera.")
        return 0

    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    names = ['', '', 'Ramjee']
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        # Show video frame
        ret_vid, frame_vid = video.read()
        if not ret_vid:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        cv2.imshow("Authentication In Progress...", frame_vid)

        # Process camera silently
        ret_cam, img = cam.read()
        if not ret_cam:
            print("Failed to capture from camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5,
            minSize=(int(minW), int(minH))
        )

        for (x, y, w, h) in faces:
            id, accuracy = recognizer.predict(gray[y:y+h, x:x+w])
            confidence = 100 - accuracy
            # print(f"Confidence: {confidence}")

            if confidence > 50:
                flag = 1
                break


        if flag == 1:
            break

        if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit early
            break

    # Clean up
    cam.release()
    video.release()
    cv2.destroyAllWindows()

    # Final message (optional)
    if flag == 1:
        print("Face matched! Access granted.")
        final_img = 255 * np.ones((300, 600, 3), dtype=np.uint8)
        cv2.putText(final_img, "FACE MATCHED!", (50, 150), font, 1.5, (0, 255, 0), 4)
        cv2.imshow("Success", final_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return flag

# AuthenticateFace()