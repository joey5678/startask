import threading
import time
import cv2
import numpy as np
from utils import get_four_points
from utils import binaryzation
from PIL import Image
import pytesseract
import keyboard


exit_mark = False
person_time_flag = False

def wait_person_finish(e):
    global person_time_flag
    return not person_time_flag
    

def manual_check_start():
    global person_time_flag
    person_time_flag = True
    print("Please check the marks manually, then click C and Z to continu.")

def manual_done(e):
    global person_time_flag
    person_time_flag = False
    print("the C is pressed and released, then press S to quit.")

def listener_time_flag():
    keyboard.on_release_key('c', manual_done)
    keyboard.wait(hotkey='s')

def print_pressed_keys(e):
    global exit_mark
    exit_mark = True
    print("the Q is pressed and released, then press ESC to quit.")


def listener_key():
    keyboard.on_release_key('q', print_pressed_keys)
    keyboard.wait(hotkey='esc')


def eval_dst_size(pts_src):

    if abs(pts_src[1][0] - pts_src[0][0]) > 1.5 * abs(pts_src[1][1] - pts_src[0][1]):
        H = abs(max(pts_src[3][1], pts_src[2][1]) - max(pts_src[0][1], pts_src[1][1]))
        W = abs(max(pts_src[1][0], pts_src[2][0]) - max(pts_src[0][0], pts_src[3][0]))
    else:
        H = abs(max(pts_src[3][0], pts_src[2][0]) - max(pts_src[0][0], pts_src[1][0]))
        W = abs(max(pts_src[1][1], pts_src[2][1]) - max(pts_src[0][1], pts_src[3][1]))
    
    return H, W


def mark(image):
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 600, 600)
    # cv2.imshow("image", image)
    global exit_mark
    # format: [{'pts': pts_src, 'text':'xxxxx'}, {....}]
    marks_results = []
    while True:

        if exit_mark:
            print("exit marked.")
            cv2.destroyAllWindows()
            exit_mark = False
            break
        pts_src = get_four_points(image)
        if pts_src is None:
            continue
        # H = abs(max(pts_src[3][1], pts_src[2][1]) - max(pts_src[0][1], pts_src[1][1]))
        # W = abs(max(pts_src[1][0], pts_src[2][0]) - max(pts_src[0][0], pts_src[3][0]))
        H, W = eval_dst_size(pts_src)
        if min(H, W) < 23:
            H, W = H*1.8, W*1.8
        size = (int(W), int(H), 3)
        im_dst = np.zeros(size, np.uint8)
        print(size)
        pts_dst = np.array(
                        [
                            [0,0],
                            [size[0] - 1, 0],
                            [size[0] - 1, size[1] -1],
                            [0, size[1] - 1 ]
                            ], dtype=float
                        )

        # Calculate the homography
        h, status = cv2.findHomography(pts_src, pts_dst)

        # Warp source image to destination
        im_dst = cv2.warpPerspective(image, h, size[0:2])
        

        cv2.imwrite('img_dst.png', im_dst)
        mean_val = cv2.mean(im_dst)
        print("mean value", end=':')
        print(mean_val)


        im_dst_0 = binaryzation(im_dst)
        # enlarge the margin
        im_dst_ex = cv2.copyMakeBorder(im_dst_0, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # execute OCR
        result_0 = pytesseract.image_to_string(Image.fromarray(im_dst_ex), lang='rus+eng')

        im_dst_1 = binaryzation(im_dst, reversed=True)
        # enlarge the margin
        im_dst_ex = cv2.copyMakeBorder(im_dst_1, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # execute OCR
        result_1 = pytesseract.image_to_string(Image.fromarray(im_dst_ex), lang='rus+eng')
        # we prefer to use result_0
        
        result = result_0 
        if len(result_0) == 0 or '\n' in result_0:
            result = result_1

        orientation = 'hor' if W > 2*H else 'ver'
        print("region location: {!r}".format(pts_src.tolist()))
        print("result: " + result)
        print("orientation: " + orientation)

        marks_results.append({'pts': pts_src.tolist(), 'text': result, 'ori': orientation})
        # Show output
        cv2.imshow("Image", im_dst_0)
        cv2.waitKey(0)
        cv2.imshow("Image", im_dst_1)
        cv2.waitKey(0)
        

    return marks_results


if __name__ == '__main__' :
    # pass
    #lk = threading.Thread(target=listener_key, args=())
    #lk.start()
    keyboard.on_release_key('q', print_pressed_keys)
    # im_src = cv2.imread("test1.jpg")
    im_src = cv2.imread("test1.png")

    marks_list = mark(im_src)
    for data in marks_list:
        print("recognize text: {} in loc: {!r}".format(data['text'], data['pts']))

    # cv2.waitKey(0)  
    #lk.join()
    time.sleep(5)

