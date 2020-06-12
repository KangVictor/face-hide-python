import face_recognition
import cv2
import numpy as np

def change_mosaic(frame):
    small_image = change_resolution(frame, 0.05, 0.05)
    return change_resolution(small_image, 20, 20)

def change_resolution(img, ratio_x, ratio_y):
    return cv2.resize(img, (0, 0), fx=ratio_x, fy=ratio_y)

def change_partial_mosaic(frame, left, top, right, bottom):
    img = change_mosaic(frame)
    f2 = frame
    for y in range(left, right):
        for x in range(top, bottom):
            f2[x, y] = img[x, y]
    return f2

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load face picture and the add encoding.
victor_image = face_recognition.load_image_file("face-photos/victor.jpg")
victor_face_encoding = face_recognition.face_encodings(victor_image)[0]

# My friend's image
junghoo_image = face_recognition.load_image_file("face-photos/junghoo.jpg")
junghoo_face_encoding = face_recognition.face_encodings(junghoo_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    victor_face_encoding,
    junghoo_face_encoding
]
known_face_names = [
    "Victor",
    "Junghoo"
]



# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # print(frame.shape)

    # Resize frame of video to 1/4 size for faster face recognition processing
    bgr_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = bgr_small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # find the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < 0.4:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Processing frames
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since it was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == "Unknown":
            # Censor the face with blur  
            frame = change_partial_mosaic(frame, left, top, right, bottom)
            # Hide face with a rectangle
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), -1)
        else:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the result
    cv2.imshow('Video', frame)

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Exit webcam
video_capture.release()
cv2.destroyAllWindows()