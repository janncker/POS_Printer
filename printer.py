#!/usr/bin/env python

import serial
import time
import cv2
import sys
import os
import numpy
import numpy as np

printer = serial.Serial("/dev/ttyUSB0", 9600)

def print_out( data):
    time.sleep(0.01)
    if printer:
        print [dd for dd in bytearray(data)]
        print printer.write(bytearray(data))
        printer.drainOutput()

def get_int32(data, offset):
    return data[offset] + int(data[offset + 1] << 8) + int(data[offset + 2] << 16) + int(data[offset + 3] << 24)


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in radians). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # fit  the POS printer
    new_w = int(new_w / 8 )*8 + (8 if new_w % 8 else 0)
    new_h = int(new_h / 8 )*8 + (8 if new_h % 8 else 0)

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue = 0xff
    )

    return result




def print_label(text_list):

    print_out([0x1b, 0x40])

    # middle
    #print_out([0x1b, 0x61, 1])
    # 2 times bigger
    print_out([0x1b, 0x21, 0x30])
    if len(text_list) == 2:
        print_out(text_list[0])
        print_out([0x0a])
        print_out(text_list[1])
        print_out([0x0a])
    elif len(text_list) == 1:
        for  i in range(30):
            print_out([0x1b, 0x4a, 1])#,0x0a, 0x0a])
        print_out(text_list[0])
        print_out([0x0a])
        for  i in range(30):
            print_out([0x1b, 0x4a, 1])#,0x0a, 0x0a])

    for  i in range(42*4):
        print_out([0x1b, 0x4a, 1])#,0x0a, 0x0a])


def print_img(data):

    datalen = get_int32(data, 0x02)
    dataoffset =  get_int32(data, 0x0a)
    width = get_int32(data, 0x12)
    height = get_int32(data, 0x16)


    mono_data = []
    
    print "0x%x" %(dataoffset)
    print "width ", width, " height ", height
    print datalen
    print dataoffset

    for index in range(dataoffset, datalen, 8):
        d7 =  0 if data[index] == 0xff else 1
        d6 = 0 if data[index + 1] == 0xff else 1
        d5 = 0 if data[index + 2] == 0xff else 1
        d4 = 0 if data[index + 3] == 0xff else 1
        d3 = 0 if data[index + 4] == 0xff else 1
        d2 = 0 if data[index + 5] == 0xff else 1
        d1 = 0 if data[index + 6] == 0xff else 1
        d = 0 if data[index + 7] == 0xff else 1
        d = d + int(d1 <<1) + int(d2 << 2) + int(d3 << 3) + int(d4 << 4) + int(d5 << 5) + int(d6 << 6) + int(d7 << 7)
        mono_data.append(d)
    print "-------------"
#
#    bmp_data = [255-bb for bb in data[dataoffset:datalen]]
#
##    print bmp_data
#
#
#    # init
    print_out([0x1b, 0x40])
#
#    # print test page
#    print_out([0x12, 0x54])
#
#    # img
    cmd = [0x1d, 0x2a, height/8, width/8]
    cmd = cmd + mono_data

    print_out(cmd)
    time.sleep(2)
    # Print with 2 times bigger
    print_out([0x1d, 0x2f, 0x3])

    print_out([0x0a])#,0x0a, 0x0a])


label_text = []


# Print picture
if os.path.isfile(sys.argv[1]):

    img = cv2.imread(sys.argv[1])
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('f1', gray)
    
    gray2 = rotate_image(gray, 90)
    
    gra = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
    
    
    cv2.imwrite("dd.bmp", gra)
    
    ret, data = cv2.imencode(".bmp", gra)
    
    cv2.imshow("dd", gra)
    cv2.waitKey(0)

    print_img(bytearray(data))

else:    #Print Label
    for t in sys.argv[1:]:
        if isinstance(t, str):
            label_text.append(t.decode("utf-8").encode('GBK'))

    print_label(label_text)
