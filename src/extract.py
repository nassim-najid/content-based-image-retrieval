import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

ORB_SIZE = (300, 300)
ORB_FEATURES = 500

def _read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Image illisible : {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-8)

def extract_color_histogram(image):
    hist = []
    for channel in range(3):
        h = cv2.calcHist([image], [channel], None, [32], [0, 256]).flatten()
        hist.extend(h)
    return _normalize(hist)

def extract_hsv_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    return _normalize(hist.flatten())

def extract_lbp_features(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 12), range=(0, 11))
    return _normalize(hist)

def extract_hu_moments(gray):
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return _normalize(-np.sign(hu) * np.log10(np.abs(hu) + 1e-12))

def extract_hog_features(gray):
    values = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    return _normalize(values)

def extract_color_moments(image):
    values = []
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        values.extend([channel.mean(), channel.std(), np.mean(np.abs(channel-channel.mean())**3)**(1/3)])
    return _normalize(values)

def extract_edge_histogram(gray):
    edges = cv2.Canny(gray, 100, 200)
    hist, _ = np.histogram(edges, bins=16, range=(0, 256))
    return _normalize(hist)

def extract_global_features(image_path):
    image = cv2.resize(_read_rgb(image_path), (300, 300))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blocks = [
        extract_color_histogram(image),
        extract_hsv_histogram(image),
        extract_lbp_features(gray),
        extract_hu_moments(gray),
        extract_hog_features(gray),
        extract_color_moments(image),
        extract_edge_histogram(gray),
    ]
    # Pondération uniforme, puis normalisation du vecteur concaténé.
    return _normalize(np.concatenate(blocks))

def extract_orb_features(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image illisible : {image_path}")
    image = cv2.resize(image, ORB_SIZE)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
    return keypoints, descriptors
