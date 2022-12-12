import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import multiprocessing as mp

from tqdm import tqdm

IMAGE_PATH = 'data/galaxy2_g.jpg'

parser = argparse.ArgumentParser()
args = parser.parse_args()


def show_image(img: np.ndarray, win_name: str = 'img'):
    img = img.astype(np.uint8)
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, img)
    cv.waitKey(0)
    cv.destroyWindow(win_name)


def overlay_image(base_img: np.ndarray, overlay_img: np.ndarray, win_name: str = 'overlay'):
    def nothing(x):
        pass
    shapes = overlay_img.copy()
    mask = shapes.astype(bool)
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.createTrackbar('alpha', win_name, 0, 100, nothing)
    alpha = 0
    while True:
        bg_img = base_img.copy()
        bg_img[mask] = cv.addWeighted(bg_img, 1 - alpha, shapes, alpha, 0)[mask]
        cv.imshow(win_name, bg_img)
        if cv.waitKey(1) & 0xFF == 27:
            break
        alpha = cv.getTrackbarPos('alpha', win_name) / 100.0

    cv.destroyWindow(win_name)
        

def get_star_positions(img: np.ndarray, step_size: int = 128):
    '''

    '''

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img3 = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
    assert gray_img.dtype == np.uint8


    _local_dark_map = np.zeros_like(gray_img)

    _local_star_map = np.zeros_like(img)
   #  _local_star_map = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

    def get_local_sky_range(local_img: np.ndarray, select_ratio: float, cut_ratio: float = 0.0) -> tuple:
        '''
        returns a tuple (dark_pixel, star_th, 255). We regard pixels as star if
        its range belong to (star_th, 255).

        dark_val  = non-star brightness value of the given patch
        cut_ratio = ratio of ignored birght pixels in the patch

            Parameters:
                local_img   : grayscale image
                select_ratio: 
            Returns:
                local_star_range which
        '''
        cut_idx = int(local_img.size * (1 - cut_ratio))
        
        local_dark_val = np.mean(np.sort(local_img.flatten())[0:cut_idx])
        local_star_th = np.uint8(local_dark_val + (255 - local_dark_val) * (1.0 - select_ratio))
        white = 255
        local_sky_range = (local_dark_val, local_star_th, white)
        return local_sky_range


    (img_h, img_w) = gray_img.shape
    # for each sub-image (patch), 
    for box_y1 in tqdm(range(0, img_h, step_size)):
        for box_x1 in range(0, img_w, step_size):

            box_y2 = min(box_y1 + step_size, img_h)
            box_x2 = min(box_x1 + step_size, img_w)

            local_img = img[box_y1:box_y2, box_x1:box_x2]

            (local_dark_val, local_star_th, white) = get_local_sky_range(local_img, select_ratio=0.75, cut_ratio=0.0)

            _local_dark_map[box_y1:box_y2, box_x1:box_x2] = local_dark_val
            
            for y in range(box_y1, box_y2 + 1):
                for x in range(box_x1, box_x2 + 1):
                    if np.all(img[y:y+2,x:x+2] > local_star_th):
                        _local_star_map[y:y+2,x:x+2] = (0,0, 255) # red

    show_image(_local_dark_map)
    show_image(_local_star_map)
    overlay_image(_local_star_map, gray_img3)

    num_labels, labels_im = cv.connectedComponents(cv.cvtColor(_local_star_map, cv.COLOR_BGR2GRAY))
    print(type(labels_im))
    print(labels_im.shape)
    print(labels.dtype)
    return labels_im


        
def main():
    img = cv.imread(IMAGE_PATH)
    show_image(img)
    get_star_positions(img, 64)




if __name__ == '__main__':
    main()
