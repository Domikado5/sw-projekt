import cv2
import numpy as np
from tensorflow.keras.models import load_model

def init():
    """
    Loading the classifier, model and initializing the attributes vector
    """
    global face_cascade
    global model
    global attributes

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = load_model('model.h5')

    attributes = np.array(['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                           'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
                           'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                           'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                           'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                           'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                           'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                           'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])


def detect_faces(image):
    """
    Face detection using the cascade classifier
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.2, 6, minSize=(50, 50))

    return faces


def crop_face(image, pos):
    """
    Cropping the face from the image
    """
    (x, y, w, h) = pos
    
    return image[y:y+h, x:x+w]


def preprocess_face(image):
    """
    Preprocessing the face in the way that's suitable for the model
    """
    prep_face = cv2.resize(image, (218, 178))
    prep_face = prep_face.reshape(-1, 218, 178, 3)  # uncomment if there are problems with dimensions

    return prep_face


def contour_face(image, face):
    """
    Drawing a rectangle around the face
    """
    (x, y, w, h) = face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


def describe_face(image, face, attrib):
    """
    Write attributes next to the face
    """
    (x, y, w, h) = face
    text = ' '.join(attrib)
    cv2.putText(image, text, (int(x+w/2 - 30), y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)



def make_prediction(face):
    """
    Making attributes prediction for a face
    Return: a list/array of attributes
    """
    pred = model.predict(face).reshape(-1)
    mask = pred > 0.5  # creating a boolean mask for attributes, the threshold can be changed
    face_attributes = attributes[mask]

    return face_attributes


def main():
    cam = cv2.VideoCapture(0)
    while True:
        _, img = cam.read()
        faces = detect_faces(img)

        for face_pos in faces:
            cropped_face = crop_face(img, face_pos)
            prep_face = preprocess_face(cropped_face)
            prediction = make_prediction(prep_face)
            contour_face(img, face_pos)
            describe_face(img, face_pos, prediction)

        cv2.imshow('img', img)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


def predict_from_photo(img_name):
    img = cv2.imread(img_name)
    faces = detect_faces(img)

    for face_pos in faces:
        cropped_face = crop_face(img, face_pos)
        prep_face = preprocess_face(cropped_face)
        prediction = make_prediction(prep_face)
        contour_face(img, face_pos)
        describe_face(img, face_pos, prediction)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    init()
    main()
    # predict_from_photo('02.png')