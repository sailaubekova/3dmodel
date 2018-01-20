import skimage
import skimage.transform
import dlib
import cv2
import numpy as np

# Dlib face detector
detector = dlib.get_frontal_face_detector()

# Dlib landmarks predictor
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# OpenCV HAAR cascade
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')


def draw_str(dst, (x, y), s):
    """
    Copy of helper function from OpenCV common.py to draw a piece of text on the top of the image

    :param dst: image where text should be added
    :param (x, y): tuple of coordinates to place text to
    :param s: string of text
    :return: image with text
    """
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


def _merge_images(img_top, img_bottom, mask=0):
    """
    Function to combine two images with mask by replacing all pixels of img_bottom which
    equals to mask by pixels from img_top.

    :param img_top: greyscale image which will replace masked pixels
    :param img_bottom: greyscale image which pixels will be replace
    :param mask: pixel value to be used as mask (int)
    :return: combined greyscale image
    """
    img_top = skimage.img_as_ubyte(img_top)

    ### convert to black
    for i in range(len(img_bottom)):
        for j in range(len(img_bottom[i])):
            img_bottom[i][j] = 0
    ###

    img_bottom = skimage.img_as_ubyte(img_bottom)
    merge_layer = img_top == mask
    img_top[merge_layer] = img_bottom[merge_layer]
    return img_top


def _shape_to_array(shape):
    """
    Function to convert dlib shape object to array

    :param shape:
    :return:
    """
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=float)


def _detect_face_dlib(image):
    """
    Function to detect faces in the input image with dlib

    :param image: grayscale image with face(s)
    :return: dlib regtangles object with detected face regions
    """
    return detector(image, 1)


def _detect_face_opencv(image, cascade):
    """
    Function to detect faces in the input image with OpenCV

    :param image: grayscale image with face(s)
    :param cascade: OpenCV CascadeClassifier object
    :return: array of detected face regions
    """
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return sorted(rects, key=lambda rect: rect[2] - rect[0], reverse=True)


    # Helpers
    output_shape = dst_face.shape[:2]  # dimensions of our final image (from webcam eg)

    # Get the landmarks/parts for the face.
    try:
        dst_face_lm = find_landmarks(dst_face, predictor, opencv_facedetector=True)
        src_face_coord = _shape_to_array(src_face_lm)
        dst_face_coord = _shape_to_array(dst_face_lm)
        warp_trans = skimage.transform.PiecewiseAffineTransform()
        warp_trans.estimate(dst_face_coord, src_face_coord)
        warped_face = skimage.transform.warp(src_face, warp_trans, output_shape=output_shape)
    except:
        warped_face = dst_face

    # Merge two images: new warped face and background of dst image
    # through using a mask (pixel level value is 0)
    warped_face = _merge_images(warped_face, dst_face)
    return warped_face
