#undo comment as required according to you

#FACE_DETECTION code

# # import cv2

# # alg = 'haarcascade_frontalface_default.xml'
# # haar_cascade = cv2.CascadeClassifier(alg)
# # cam = cv2.VideoCapture(0)

# # while True:
# #     _,img = cam.read()
# #     text="Face not detected"
# #     grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #     face = haar_cascade.detectMultiScale(grayImg,1.3,4)
# #     for (x,y,w,h) in face:
# #         text="Face Detected"
# #         cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)
# #     print(text)
# #     image = cv2.putText(img, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
# #     cv2.imshow("Face Detection", image)
# #     key = cv2.waitKey(10)
# #     if key == 27:
# #         break

# # cam.release()
# # cv2.destroyAllWindows()

# Comment the all code except above one to run above code

#SHAPE_OF_FACE


# import cv2

# # Load the pre-trained face cascade classifier
# face_cascade = cv2.CascadeClassifier(r"C:\Users\gaurk\OneDrive\Desktop\Face_Detection wopencv\Face-Detection-Using-Python\haarcascade_frontalface_default.xml")

# # Define the face shape labels
# face_shape_labels = ['Round', 'Heart', 'Square', 'Oval', 'long']

# def detect_face_shape(face):
#     # Calculate the aspect ratio of the detected face
#     (x, y, w, h) = face
#     aspect_ratio = float(w) / h

#     # Classify the face shape based on the aspect ratio
#     if aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
#         return 'Round'
#     elif aspect_ratio >= 0.7 and aspect_ratio <= 0.9:
#         return 'Heart'
#     elif aspect_ratio >= 1.1 and aspect_ratio <= 1.3:
#         return 'Square'
#     else:
#         return 'Oval'

# # Open the video capture
# cam = cv2.VideoCapture(0)

# while True:
#     # Read the frame from the video capture
#     _, img = cam.read()

#     # Convert the frame to grayscale
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale image
#     faces = face_cascade.detectMultiScale(gray_img, 1.3, 4)

#     # Process each detected face
#     for (x, y, w, h) in faces:
#         # Detect the face shape
#         face_shape = detect_face_shape((x, y, w, h))

#         # Draw a rectangle around the face
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Display the face shape label
#         cv2.putText(img, face_shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the resulting image
#     cv2.imshow("Face Detection", img)

#     # Check for the 'Esc' key to exit
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# # Release the video capture and close the OpenCV windows
# cam.release()
# cv2.destroyAllWindows()







# #Funny face code

# importing libs 

import cv2      #for img processing
import numpy as np      #for maths calculations

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(r"C:\Users\gaurk\OneDrive\Desktop\Face_Detection wopencv\Face-Detection-Using-Python\haarcascade_frontalface_default.xml")

# Open the video capture
cam = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    _, img = cam.read()

    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the radius of the round face
        radius = max(w, h) // 2

        # Create a mask for the round face
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)

        # Apply the mask to the original image
        masked_img = cv2.bitwise_and(img, mask)

        # Draw a filled circle on the round face
        cv2.circle(masked_img, (center_x, center_y), radius, (0, 255, 0), -1)

        # Overlay the round face on the original image
        img = cv2.addWeighted(img, 0.8, masked_img, 0.2, 0)

        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Round Face", img)

    # Check for the 'Esc' key to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the video capture and close the OpenCV windows
cam.release()
cv2.destroyAllWindows()     #destroys all windows



