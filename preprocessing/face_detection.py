import os
import cv2
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier("/Users/niklioni/Desktop/Artificial Intelligence/Assessment/haarcascade_frontalface_default.xml")

def extract_faces(img, face_cascade):
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return [img[y:y+h, x:x+w] for (x, y, w, h) in faces]



def load_and_detect_faces_jaffe(folder_path, return_filenames=False):
    images = []
    labels = []
    fnames = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces = extract_faces(img, face_cascade)
        for face in faces:
            images.append(face)
            labels.append(filename[3:5])   # AN, DI, FE, HA, undso
            fnames.append(filename)

    if return_filenames:
        return images, labels, fnames
    return images, labels

def load_and_detect_faces_ck(folder_path):
    images = []
    labels = []

    for label in os.listdir(folder_path):
        label_dir = os.path.join(folder_path, label)
        if not os.path.isdir(label_dir):
            continue

        for filename in os.listdir(label_dir):
            img_path = os.path.join(label_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces = extract_faces(img, face_cascade)
            for face in faces:
                images.append(face)
                labels.append(label)

    return images, labels


def show_faces(images, labels, n=9):
    plt.figure(figsize=(8, 8))
    for i in range(min(n, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        plt.title(labels[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
