from utils import read_image, show_image, save_image
from core import DetectionDoctored


if __name__ == '__main__':
    img = read_image('imgs/planes.jpg')
    dd = DetectionDoctored(img)
    score= dd.run()
    show_image(score)
    save_image(score, 'results/anomaly_score.png')