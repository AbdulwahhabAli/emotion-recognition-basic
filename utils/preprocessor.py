import numpy as np
import cv2


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def _imread(image_name):
        # Use OpenCV to read image and convert BGR->RGB to match imageio behavior
        img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_name}")
        # If image has 3 channels, convert BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

def _imresize(image_array, size):
        # size is expected as (height, width) in this repository
        try:
            h, w = size
        except Exception:
            raise ValueError(f"size must be (height, width), got: {size}")
        # OpenCV expects size as (width, height)
        target = (int(w), int(h))
        # cv2.resize expects numpy array with shape (H,W[,C]) and returns same
        # Use INTER_LINEAR (BILINEAR) which is similar to scipy.misc.imresize default
        return cv2.resize(image_array, target, interpolation=cv2.INTER_LINEAR)

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical

