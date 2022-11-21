import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import multiprocessing as mp

from tqdm import tqdm

IMAGE_PATH = '../data/galaxy3.ppm'

parser = argparse.ArgumentParser()
args = parser.parse_args()


def show_image(img: np.ndarray, win_name: str = 'img'):
    img = img.astype(np.uint8)
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, img)
    cv.waitKey(0)
    cv.destroyWindow(win_name)


def get_star_positions(img: np.ndarray, step_size: int = 128):
    '''

    '''

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    assert gray_img.dtype == np.uint8

    _local_dark_map = np.zeros_like(gray_img)



    def get_local_sky_range(local_img: np.ndarray, alpha: float = 0.75) -> tuple:
        '''
        returns a tuple (dark_pixel, star_th, 255). We regard pixels as star if
        its range belong to (star_th, 255).

            Parameters:
                local_img   : grayscale image
                alpha       : 
            Returns:
                local_star_range which 
        '''
        local_dark_val = np.mean(local_img)
        local_star_th = np.uint8(local_dark_val + (255 - local_dark_val) * (1.0 - alpha))
        white = 255
        local_sky_range = (local_dark_val, local_star_th, white)
        return local_sky_range


    (img_h, img_w) = gray_img.shape

    for box_y1 in range(0, img_h, step_size):
        for box_x1 in range(0, img_w, step_size):

            box_y2 = box_y1 + step_size
            box_x2 = box_x1 + step_size

            local_img = img[box_y1:box_y2, box_x1:box_x2]

            (local_dark_val, local_star_th, white) = get_local_sky_range(local_img)
            
            _local_dark_map[box_y1:box_y2, box_x1:box_x2] = local_dark_val

    show_image(_local_dark_map)



        



def main():
    img = cv.imread(IMAGE_PATH)
    show_image(img)
    get_star_positions(img)




if __name__ == '__main__':
    main()
    exit(0)
