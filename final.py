import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

def show_image_processing():
    img = cv2.imread('fish.jpg')

    # Convert to RGB and display the original image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.show()

    # Convert to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Laplace filter
    laplician_img = cv2.Laplacian(grayscale, cv2.CV_64F)
    plt.imshow(laplician_img, "gray")
    plt.title("Laplace Filter")
    plt.show()

    # Sobel X filter
    sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, 3)
    plt.imshow(sobel_x, "gray")
    plt.title("Sobel X Filter")
    plt.show()

    # Sobel Y filter
    sobel_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, 3)
    plt.imshow(sobel_y, "gray")
    plt.title("Sobel Y Filter")
    plt.show()

    # Canny Edge Detection
    canny = cv2.Canny(grayscale, 100, 200)
    plt.imshow(canny, "gray")
    plt.title("Canny Edge Detection")
    plt.show()

def show_edge_detection():
    img = cv2.imread('fish.jpg', 0)

    # Smoothing filters
    mean_img = cv2.blur(img, (11, 11))
    median_img = cv2.medianBlur(img, 11)
    gaussian_img = cv2.GaussianBlur(img, (11, 11), 0)
    bilateral_img = cv2.bilateralFilter(img, 11, 120, 120)

    result_img = [mean_img, median_img, gaussian_img, bilateral_img]
    result_title = ['Mean', 'Median', 'Gaussian', 'Bilateral']

    # Display smoothing results
    for idx, (img, title) in enumerate(zip(result_img, result_title)):
        plt.subplot(2, 2, idx+1)
        plt.imshow(img, 'gray')
        plt.title(title)
    plt.show()

    # Thresholding techniques
    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, img_binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    _, img_tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    _, img_tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    _, img_trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    _, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    result_img = [img_binary, img_binary_inv, img_tozero, img_tozero_inv, img_trunc, img_otsu]
    result_title = ['Binary', 'Binary Inverse', 'Tozero', 'Tozero Inverse', 'Truncate', 'Otsu']

    # Display thresholding results
    for idx, (img, title) in enumerate(zip(result_img, result_title)):
        plt.subplot(2, 3, idx+1)
        plt.imshow(img, 'gray')
        plt.title(title)
    plt.show()

def show_face_recognition():
    train_path = 'train'
    person_names = os.listdir(train_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_list = []
    class_list = []

    # Training
    for index, person_name in enumerate(person_names):
        full_name_path = os.path.join(train_path, person_name)
        for image_path in os.listdir(full_name_path):
            full_image_path = os.path.join(full_name_path, image_path)
            img_gray = cv2.imread(full_image_path, 0)

            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            if len(detected_faces) < 1:
                continue

            for face_rect in detected_faces:
                x, y, w, h = face_rect
                face_img = img_gray[y:y + w, x:x + h]

                face_list.append(face_img)
                class_list.append(index)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(class_list))

    # Testing
    test_path = 'test'
    for image_path in os.listdir(test_path):
        full_image_path = os.path.join(test_path, image_path)
        img_bgr = cv2.imread(full_image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        if len(detected_faces) < 1:
            continue

        for face_rect in detected_faces:
            x, y, w, h = face_rect
            face_img = img_gray[y:y + w, x:x + h]

            res, confidence = face_recognizer.predict(face_img)
            confidence = math.floor(confidence * 100) / 100

            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)
            text = f'{person_names[res]} {confidence}%'
            cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

            cv2.imshow('Face Recognition Result', img_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def show_shape_detection():
    img_object = cv2.imread('clownfish.png')
    img_scene = cv2.imread('clownfishframe2.jpg')

    # Use ORB instead of SURF
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors with ORB
    kp_object, des_object = orb.detectAndCompute(img_object, None)
    kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

    # FLANN parameters for ORB
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_object, des_scene, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    total_match = 0
    for i, match in enumerate(matches):
        if len(match) == 2:  # Ensure there are 2 matches to unpack
            m, n = match
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                total_match += 1

    img_res = cv2.drawMatchesKnn(
        img_object, kp_object, img_scene,
        kp_scene, matches, None,
        matchColor=[0, 255, 0],
        singlePointColor=[255, 0, 0],
        matchesMask=matchesMask
    )
    plt.imshow(img_res)
    plt.show()

def menu():
    while True:
        print("\nMenu:")
        print("1. Show Image Processing")
        print("2. Show Smoothing and Thresholding")
        print("3. Show Face Recognition")
        print("4. Show Shape Detection")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            show_image_processing()
        elif choice == '2':
            show_edge_detection()
        elif choice == '3':
            show_face_recognition()
        elif choice == '4':
            show_shape_detection()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    menu()