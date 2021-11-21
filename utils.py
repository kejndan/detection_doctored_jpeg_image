from cv2 import cv2
import matplotlib.pyplot as plt

def read_image(path):
    rgb_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    return rgb_image

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def save_image(img, name):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(name)


def split_into_blocks(matrix, size):
    h, w = matrix.shape
    return matrix.reshape(h//size[0], size[0], w//size[1], size[1]).swapaxes(1,2).reshape(-1, size[0], size[1])