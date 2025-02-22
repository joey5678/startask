import threading
import os
import sys
import time
import uuid
from io import BytesIO

from PIL import Image
import cv2
import numpy as np
import keyboard
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait

from utils import get_four_points
from utils import binaryzation
from utils import eval_dst_size

# from ocr_demo import mark


x_flag = False #exit whole program flag
c_flag = False #screenshot flag
s_flag = False #submit flag
a_flag = False #let automatication continue flag
w_flag = True # wait flag. If true, wait, otherwise, go. 
q_flag = True # notify the marking on image is done, close the opencv image and return to page.


def reset_flag():
    global a_flag, c_flag, s_flag, x_flag, w_flag, q_flag
    x_flag = False 
    c_flag = False 
    s_flag = False 
    a_flag = False
    w_flag = True 
    q_flag = False


def print_pressed_keys(e):
    global a_flag, c_flag, s_flag, x_flag, w_flag
    print("the {} is pressed and released.".format(e.name))

    if e.name == 'a':
        a_flag = True
    elif e.name == 'c':
        c_flag = True
    elif e.name == 's':
        s_flag = True
    elif e.name == 'x':
        x_flag = True
    elif e.name == 'enter':
        w_flag = False
    elif e.name == 'q':
        q_flag = True


def keyb_wait():
    keyboard.on_release_key('a', print_pressed_keys)
    keyboard.on_release_key('c', print_pressed_keys)
    keyboard.on_release_key('s', print_pressed_keys)
    keyboard.on_release_key('x', print_pressed_keys) 
    keyboard.on_release_key('enter', print_pressed_keys) 
    keyboard.on_release_key('q', print_pressed_keys)


def listen_waiting_finish(e):
    global w_flag, c_flag, s_flag
    return not w_flag and (c_flag or s_flag or x_flag)


def login_in(driver, user='18665860156', pwd='Aa,123.456'):
    # driver = webdriver.Chrome()
    driver.get("https://task.startask.net")

    pinput = driver.find_element_by_name("phone")
    pwdinput = driver.find_element_by_name("password")

    pinput.send_keys(user)
    pwdinput.send_keys(pwd)

    btn = driver.find_elements_by_tag_name('button')
    assert len(btn) == 1
    btn = btn[0]
    btn.click()

    ime.sleep(2)
    # link_ele = driver.find_element_by_xpath('/*[@id="root"]/div/div/div/div/div[2]/div/a')
    link_ele = driver.find_element_by_class_name('sc-kafWEX')
    link_ele.click()
    time.sleep(3)

    btn = driver.find_element_by_xpath('//*[@id="root"]/div/div/div/div/div[2]/div/div[1]/div[2]/button')
    # btn = btn[0]
    btn.click()

    time.sleep(2)

    btn = driver.find_element_by_xpath('//*[@id="root"]/div/div/div/div/div/div[3]/button')
    btn.click()
    time.sleep(5) # enter the marking page.
    return True


def wait_user(driver):
    global w_flag
    WebDriverWait(driver, 999999).until(listen_waiting_finish)
    w_flag = True # reset wflag.


def is_screenshot():
    global c_flag
    if c_flag:
        c_flag = False
        return True
    return c_flag


def is_submit():
    global s_flag
    if s_flag:
        s_flag = False
        return True
    return s_flag


def is_exit():
    global x_flag
    if x_flag:
        x_flag = True
        return True
    return x_flag


def mark(image):

    global q_flag
    marks_results = []
    while True:
        if q_flag:
            print("exit marked.")
            cv2.destroyAllWindows()
            q_flag = False
            break
        pts_src = get_four_points(image)
        if pts_src is None:
            continue

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

        h, status = cv2.findHomography(pts_src, pts_dst)

        # Warp source image to destination
        im_dst = cv2.warpPerspective(image, h, size[0:2])
        
        # cv2.imwrite('img_dst.png', im_dst)
        # mean_val = cv2.mean(im_dst)
        # print("mean value", end=':')
        # print(mean_val)

        im_dst_0 = binaryzation(im_dst)
        im_dst_ex = cv2.copyMakeBorder(im_dst_0, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        result_0 = pytesseract.image_to_string(Image.fromarray(im_dst_ex), lang='rus+eng')

        im_dst_1 = binaryzation(im_dst, reversed=True)
        im_dst_ex = cv2.copyMakeBorder(im_dst_1, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        result_1 = pytesseract.image_to_string(Image.fromarray(im_dst_ex), lang='rus+eng')

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


def auto_screenshot(driver):
    cv_ele = driver.find_element_by_class_name('canvas-container')
    cv_png = cv_ele.screenshot_as_png
    gen_name = 'RUS_{}'.format(str(uuid.uuid1()))
    cv_ele.screenshot('{}.png'.format(gen_name))
    ss = Image.open(BytesIO(cv_png))
    img = cv2.cvtColor(np.asarray(ss),cv2.COLOR_RGB2BGR)
    # wait the image screenshot display, then mark on the image, when complete, press 'q'.
    mark_list = mark(img)
    # save_marks_to_file(mark_list, gen_name)
    for data in mark_list:
        print("recognize text: {} in loc: {!r}".format(data['text'], data['pts']))

        ActionChains(driver).send_keys('q').perform()
        for x, y in data['pts']:
            ActionChains(driver).move_to_element_with_offset(cv_ele, x, y).click().perform()
        ActionChains(driver).send_keys('1').pause(0.5).perform()
        textarea = driver.find_element_by_tag_name('textarea')
        textarea.send_keys(data['text'])
        cls_div = driver.find_element_by_xpath('//*[@id="root"]/div/div/div/div[1]/div/div[2]/div/div[2]/div/div[2]/div[1]/div/div')

        # so far, only select the first option in list.
        ActionChains(driver).move_to_element(cls_div).click().pause(0.5).move_by_offset(0, 20).click().perform()
    
    print("auto_screenshot done.")
    return True


def auto_submit(driver):
    smb_btn = driver.find_element_by_xpath('//*[@id="root"]/div/div/footer/div[2]/button')
    smb_btn.click()
    time.sleep(0.5)
    cfm_btn = driver.find_element_by_xpath('//*[@id="react-confirm-alert"]/div/div/div/div/button[1]')
    cfm_btn.click()
    time.sleep(5)


def run_flow(driver):
    login_in(driver)
    wait_user(driver)
    if is_screenshot():
        try:
            auto_screenshot(driver)
        except Exception as e:
            print("some exception in screenshot and marking: {}".format(str(e)))
    elif is_submit():
        try:
            auto_submit(driver)
        except Exception as e:
            print("some exception in submit and marking: {}".format(str(e)))
    elif is_exit():
        sys.exit(0)

def run_flow_mock(driver):
    # login_in(driver)
    print("Login in ....")

    while True:
        wait_user(driver)
        if is_screenshot():
            try:
                print("auto screenshot....")
                # auto_screenshot(driver)
            except Exception as e:
                print("some exception in screenshot and marking: {}".format(str(e)))
        elif is_submit():
            try:
                # auto_submit(driver)
                print("auto_submit...")
            except Exception as e:
                print("some exception in submit and marking: {}".format(str(e)))
        elif is_exit():
            sys.exit(0)

if __name__ == "__main__":
    reset_flag()
    keyb_wait()
    # driver = webdriver.Chrome()
    driver = None
    run_flow_mock(driver)