# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections
import glob
import pickle
import random
import tensorflow as tf
import argparse
from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
import os.path
import time

from unet import UNet
from utils import compile_frames_to_gif

# 글씨 있는 부분 좌표 추출
def crop_image(img):
    image = np.array(img)
    blur = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=0)
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blur, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    contours_xy = np.array(contours)
    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
      for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0])
        x_min = min(value)
        x_max = max(value)
 
    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
      for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1])
        y_min = min(value)
        y_max = max(value)

    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min

    return x, y, w, h

# 글씨 있는 부분으로만 자르고 비율로 리사이즈
def process_image(img, x, y, w, h, canvas_size):
    new_width = int(canvas_size * 0.5)
    new_height = int(new_width * h / w)
    if new_height > canvas_size - 10:
        new_height = int(canvas_size * 0.6)
        new_width = int(new_height * w / h)
    img = img.crop((x-1, y-1, x+w+1, y+h+1)).resize((new_width,new_height))
    new_left = int((canvas_size - img.width) / 2)
    new_top = int((canvas_size - img.height) / 2)
    result = Image.new("L", (canvas_size, canvas_size), color=255)
    result.paste(img, (new_left, new_top))

    return result

def make_handwriting_image(full_img, canvas_size, count):
    width, height = 135, 135
    x, y = 73, 146
    diff = 208

    startx, endx = x+width*(count%15), x+width*(count%15+1)
    starty, endy = y+diff*(count//15), y+diff*(count//15)+height
    image = full_img.crop((startx, starty, endx, endy))
    x, y, w, h = crop_image(image)
    image = process_image(image, x-5, y-5, w+10, h+10, canvas_size)
    return image

def draw_single_char(st, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), st, (0, 0, 0), font=font)
    x, y, w, h = crop_image(img)
    img = process_image(img, x, y, w, h, canvas_size)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset, full_img, count):
    dst_img = make_handwriting_image(full_img, canvas_size, count)
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img

def font2img(src, stringset, char_size, canvas_size, x_offset, y_offset, sample_dir, label, full_img):
    src_font = ImageFont.truetype(src, size=char_size)
    count = 0
    for s in stringset:
        e = draw_example(' '+s, src_font, canvas_size, x_offset, y_offset, full_img, count)
        if e:
            e.save(os.path.join(sample_dir, "%02d_%04d.jpg" % (label, count)))
            count += 1

# train.obj와 val.obj 생성
def pickle_examples(paths, train_path, val_path, train_val_split):
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for p in paths:
                label = int(os.path.basename(p).split("_")[0])
                with open(p, 'rb') as f:
                    print("img %s" % p, label)
                    img_bytes = f.read()
                    r = random.random()
                    example = (label, img_bytes)
                    if r < train_val_split:
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)

# test.obj 생성
def pickle_tests(paths, test_path):
    with open(test_path, 'wb') as fv:
        for p in paths:
            label = int(os.path.basename(p).split("_")[0])
            with open(p, 'rb') as f:
                print("img %s" % p, label)
                img_bytes = f.read()
                example = (label, img_bytes)
                pickle.dump(example, fv)


import sys
import logging
import operator
from collections import deque
from io import StringIO
from optparse import OptionParser

logging.basicConfig()
log = logging.getLogger('png2svg')

def add_tuple(a, b):
    return tuple(map(operator.add, a, b))

def sub_tuple(a, b):
    return tuple(map(operator.sub, a, b))

def neg_tuple(a):
    return tuple(map(operator.neg, a))

def direction(edge):
    return sub_tuple(edge[1], edge[0])

def magnitude(a):
    return int(pow(pow(a[0], 2) + pow(a[1], 2), .5))

def normalize(a):
    mag = magnitude(a)
    assert mag > 0, "Cannot normalize a zero-length vector"
    return tuple(map(operator.truediv, a, [mag]*len(a)))
    

def svg_header(width, height):
    return """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="%d" height="%d"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
""" % (width, height)

def rgba_image_to_svg_pixels(im, opaque=None):
    s = StringIO()
    s.write(svg_header(*im.size))

    width, height = im.size
    for x in range(width):
        for y in range(height):
            here = (x, y)
            rgba = im.getpixel(here)
            if opaque and not rgba[3]:
                continue
            s.write("""  <rect x="%d" y="%d" width="1" height="1" style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n""" % (x, y, rgba[0:3], float(rgba[3]) / 255))
    s.write("""</svg>\n""")
    return s.getvalue()


def joined_edges(assorted_edges, keep_every_point=False):
    pieces = []
    piece = []
    directions = deque([
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        ])
    while assorted_edges:
        if not piece:
            piece.append(assorted_edges.pop())
        current_direction = normalize(direction(piece[-1]))
        while current_direction != directions[2]:
            directions.rotate()
        for i in range(1, 4):
            next_end = add_tuple(piece[-1][1], directions[i])
            next_edge = (piece[-1][1], next_end)
            if next_edge in assorted_edges:
                assorted_edges.remove(next_edge)
                if i == 2 and not keep_every_point:
                    # same direction
                    piece[-1] = (piece[-1][0], next_edge[1])
                else:
                    piece.append(next_edge)
                if piece[0][0] == piece[-1][1]:
                    if not keep_every_point and normalize(direction(piece[0])) == normalize(direction(piece[-1])):
                        piece[-1] = (piece[-1][0], piece.pop(0)[1])
                        # same direction
                    pieces.append(piece)
                    piece = []
                break
        else:
            raise Exception("Failed to find connecting edge")
    return pieces


def rgba_image_to_svg_contiguous(im, opaque=None, keep_every_point=False):    
    adjacent = ((1, 0), (0, 1), (-1, 0), (0, -1))
    visited = Image.new("1", im.size, 0)
    
    color_pixel_lists = {}

    width, height = im.size
    for x in range(width):
        for y in range(height):
            here = (x, y)
            if visited.getpixel(here):
                continue
            rgba = im.getpixel((x, y))
            if opaque and not rgba[3]:
                continue
            piece = []
            queue = [here]
            visited.putpixel(here, 1)
            while queue:
                here = queue.pop()
                for offset in adjacent:
                    neighbour = add_tuple(here, offset)
                    if not (0 <= neighbour[0] < width) or not (0 <= neighbour[1] < height):
                        continue
                    if visited.getpixel(neighbour):
                        continue
                    neighbour_rgba = im.getpixel(neighbour)
                    if neighbour_rgba != rgba:
                        continue
                    queue.append(neighbour)
                    visited.putpixel(neighbour, 1)
                piece.append(here)

            if not rgba in color_pixel_lists:
                color_pixel_lists[rgba] = []
            color_pixel_lists[rgba].append(piece)

    del adjacent
    del visited

    edges = {
        (-1, 0):((0, 0), (0, 1)),
        (0, 1):((0, 1), (1, 1)),
        (1, 0):((1, 1), (1, 0)),
        (0, -1):((1, 0), (0, 0)),
        }
            
    color_edge_lists = {}

    for rgba, pieces in color_pixel_lists.items():
        for piece_pixel_list in pieces:
            edge_set = set([])
            for coord in piece_pixel_list:
                for offset, (start_offset, end_offset) in edges.items():
                    neighbour = add_tuple(coord, offset)
                    start = add_tuple(coord, start_offset)
                    end = add_tuple(coord, end_offset)
                    edge = (start, end)
                    if neighbour in piece_pixel_list:
                        continue
                    edge_set.add(edge)
            if not rgba in color_edge_lists:
                color_edge_lists[rgba] = []
            color_edge_lists[rgba].append(edge_set)

    del color_pixel_lists
    del edges

    color_joined_pieces = {}

    for color, pieces in color_edge_lists.items():
        color_joined_pieces[color] = []
        for assorted_edges in pieces:
            color_joined_pieces[color].append(joined_edges(assorted_edges, keep_every_point))

    s = StringIO()
    s.write(svg_header(*im.size))

    for color, shapes in color_joined_pieces.items():
        for shape in shapes:
            s.write(""" <path d=" """)
            for sub_shape in shape:
                here = sub_shape.pop(0)[0]
                s.write(""" M %d,%d """ % here)
                for edge in sub_shape:
                    here = edge[0]
                    s.write(""" L %d,%d """ % here)
                s.write(""" Z """)
            s.write(""" " style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n""" % (color[0:3], float(color[3]) / 255))
            
    s.write("""</svg>\n""")
    return s.getvalue()
    
                
def png_to_svg(filename, contiguous=None, opaque=None, keep_every_point=None):
    try:
        im = Image.open(filename)
    except IOError as e:
        sys.stderr.write('%s: Could not open as image file\n' % filename)
        sys.exit(1)
    im_rgba = im.convert('RGBA')
    
    if contiguous:
        return rgba_image_to_svg_contiguous(im_rgba, opaque, keep_every_point)
    else:
        return rgba_image_to_svg_pixels(im_rgba, opaque)


def create_handwriting_dataset(file):
    src_font = 'runmodel/source/source_font.ttf'
    f = open("runmodel/source/chosen_hangul.txt", 'r', encoding='UTF8')
    charset = f.readlines()
    char_size, canvas_size = 130, 256
    x_offset, y_offset = 20, 20
    sample_dir = 'runmodel/source/samples/'
    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)
    experiment_dir = 'runmodel/source/experiment/'
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
    save_dir = 'runmodel/source/experiment/data/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    split_ratio = 0.1
    label = 52
    
    full_img = Image.open('media/'+str(file))

    font2img(src_font, charset, char_size, canvas_size, x_offset, y_offset, sample_dir, label, full_img)

    train_path = os.path.join(save_dir, "train.obj")
    val_path = os.path.join(save_dir, "val.obj")
    pickle_examples(sorted(glob.glob(os.path.join(sample_dir, "*.jpg"))), train_path=train_path, val_path=val_path, train_val_split=split_ratio)


def train_handwriting():
    experiment_dir = 'runmodel/source/experiment/'

    experiment_id = 0
    image_size = 256

    L1_penalty, Lconst_penalty, Ltv_penalty, Lcategory_penalty = 100, 15, 0.0, 1.0
    embedding_num, embedding_dim = 53, 2350

    epoch = 200
    batch_size = 16
    lr = 0.0005
    sample_steps, checkpoint_steps = 3000, 1000

    schedule = 10
    resume = 1
    freeze_encoder = True
    fine_tune = None
    inst_norm = 0
    flip_labels = None

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        model = UNet(experiment_dir, batch_size=batch_size, experiment_id=experiment_id,
                    input_width=image_size, output_width=image_size, embedding_num=embedding_num,
                    embedding_dim=embedding_dim, L1_penalty=L1_penalty, Lconst_penalty=Lconst_penalty,
                    Ltv_penalty=Ltv_penalty, Lcategory_penalty=Lcategory_penalty)
        model.register_session(sess)
        if flip_labels:
            model.build_model(is_training=True, inst_norm=inst_norm, no_target_source=True)
        else:
            model.build_model(is_training=True, inst_norm=inst_norm)
        fine_tune_list = None
        if fine_tune:
            ids = fine_tune.split(",")
            fine_tune_list = set([int(i) for i in ids])
        model.train(lr=lr, epoch=epoch, resume=resume,
                    schedule=schedule, freeze_encoder=freeze_encoder, fine_tune=fine_tune_list,
                    sample_steps=sample_steps, checkpoint_steps=checkpoint_steps,
                    flip_labels=flip_labels)


def infer_handwriting():
    model_dir = 'runmodel/source/experiment/checkpoint/experiment_0_batch_16/'
    source_obj = 'runmodel/source/test.obj'
    save_dir = 'runmodel/source/infer/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    batch_size = 235

    embedding_num, embedding_dim = 53, 2350
    embedding_ids = '52'
    count = 0

    inst_norm = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=batch_size, embedding_num=embedding_num, embedding_dim=embedding_dim)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=inst_norm)
        embedding_ids = [int(i) for i in embedding_ids.split(",")]
        embedding_ids = embedding_ids[0]
        model.infer(model_dir=model_dir, source_obj=source_obj, embedding_ids=embedding_ids, save_dir=save_dir, count=count)


def create_svg_files():
    save_dir = 'runmodel/source/infer/letters/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    width, height = 256, 256

    img = Image.open('runmodel/source/infer/handwriting_name9.png')

    for count in range(2350):
        # 글자 별로 자르기
        startx, endx = width*(count//235), width*(count//235+1)
        starty, endy = height*(count%235), height*(count%235+1)
        image = img.crop((startx, starty, endx, endy))

        # 배경 투명하게
        rgba = image.convert("RGBA")
        datas = rgba.getdata()
        newData = []
        for item in datas:
            if item[0] > 100 and item[1] > 100 and item[2] > 100:
                newData.append((255, 255, 255, 0))
            else:
                newData.append((0, 0, 0, 255))
        rgba.putdata(newData)

        # 글자 있는 부분만 자르기
        x, y, w, h = crop_image(rgba)
        rgba = rgba.crop((x, y, x+w, y+h))

        rgba.save(save_dir+str(count)+'.png')

    svg_dir = 'runmodel/source/infer/svg/'
    if not os.path.isdir(svg_dir):
        os.mkdir(svg_dir)

    f = open("runmodel/source/2350-common-hangul.txt", 'r', encoding='cp949')
    charset = f.readlines()
    for i in range(2350):
        txt = png_to_svg('runmodel/source/infer/letters/'+str(i)+'.png', contiguous=True, opaque=True, keep_every_point=None)
        with open('runmodel/source/infer/svg/'+charset[i][0]+'.txt', "w") as file:
            file.write(txt)

    files = glob.glob("runmodel/source/infer/svg/*.txt")

    for x in files:
        if not os.path.isdir(x):
            filename = os.path.splitext(x)
            try:
                os.rename(x,filename[0] + '.svg')
            except:
                pass


JS_DROP_FILES = "var c=arguments,b=c[0],k=c[1];c=c[2];for(var d=b.ownerDocument||document,l=0;;){var e=b.getBoundingClientRect(),g=e.left+(k||e.width/2),h=e.top+(c||e.height/2),f=d.elementFromPoint(g,h);if(f&&b.contains(f))break;if(1<++l)throw b=Error('Element not interactable'),b.code=15,b;b.scrollIntoView({behavior:'instant',block:'center',inline:'center'})}var a=d.createElement('INPUT');a.setAttribute('type','file');a.setAttribute('multiple','');a.setAttribute('style','position:fixed;z-index:2147483647;left:0;top:0;');a.onchange=function(b){a.parentElement.removeChild(a);b.stopPropagation();var c={constructor:DataTransfer,effectAllowed:'all',dropEffect:'none',types:['Files'],files:a.files,setData:function(){},getData:function(){},clearData:function(){},setDragImage:function(){}};window.DataTransferItemList&&(c.items=Object.setPrototypeOf(Array.prototype.map.call(a.files,function(a){return{constructor:DataTransferItem,kind:'file',type:a.type,getAsFile:function(){return a},getAsString:function(b){var c=new FileReader;c.onload=function(a){b(a.target.result)};c.readAsText(a)}}}),{constructor:DataTransferItemList,add:function(){},clear:function(){},remove:function(){}}));['dragenter','dragover','drop'].forEach(function(a){var b=d.createEvent('DragEvent');b.initMouseEvent(a,!0,!0,d.defaultView,0,0,0,g,h,!1,!1,!1,!1,0,null);Object.setPrototypeOf(b,null);b.dataTransfer=c;Object.setPrototypeOf(b,DragEvent.prototype);f.dispatchEvent(b)})};d.documentElement.appendChild(a);a.getBoundingClientRect();return a;"

def drop_files(element, files, offsetX=0, offsetY=0):
    driver = element.parent
    isLocal = not driver._is_remote or '127.0.0.1' in driver.command_executor._url
    paths = []

    for file in (files if isinstance(files, list) else [files]) :
        if not os.path.isfile(file) :
            raise FileNotFoundError(file)
        paths.append(file if isLocal else element._upload(file))
    
    value = '\n'.join(paths)
    elm_input = driver.execute_script(JS_DROP_FILES, element, offsetX, offsetY)
    elm_input._execute('sendKeysToElement', {'value': [value], 'text': value})


def download_fontello():
    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    # 창 보이게 / 안보이게
    driver = webdriver.Chrome(executable_path="D:\program\chromedriver.exe")
    # driver = webdriver.Chrome(executable_path="D:\program\chromedriver.exe", options=options)

    driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
    params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': 'C:\\Users\\YULA\\Downloads'}}
    command_result = driver.execute("send_command", params)

    url = 'https://fontello.com/'
    driver.get(url)

    WebElement.drop_files = drop_files
    driver.implicitly_wait(10)
    dropzone = driver.find_element_by_css_selector("#custom_icons > div > div")

    path = 'D:/handwriting-server/runmodel/source/infer/chosen'
    files = os.listdir(path)
    file_list = []
    for f in files:
        file_list.append(path+'/'+f)

    dropzone.drop_files(file_list)
    driver.implicitly_wait(3000)

    driver.execute_script("window.scrollTo(0, 0)")

    for n in range(len(file_list)):
        if n%16 == 0:
            height = 55*((n//16)+1)
            driver.execute_script("window.scrollTo(0, %s);" % height)
            time.sleep(2)
        # 글자 선택
        select = driver.find_element_by_css_selector("#custom_icons > div > ul > li:nth-child({0}) > div".format(n+1))
        select.click()
        # 수정 모드
        edit = driver.find_element_by_css_selector("#custom_icons > div > ul > li:nth-child({0}) > a".format(n+1))
        edit.click()
        # 무슨 글자인지 받아옴
        word = driver.find_element_by_css_selector("#gopt__css_name")
        keyboard = driver.find_element_by_xpath('//*[@id="gopt__code"]')
        # 입력키 변경
        keyboard.clear()
        keyboard.send_keys(hex(ord(word.get_attribute('value')))[2:])
        # 저장
        save = driver.find_element_by_css_selector("body > div.modal.fade.show > div > div > div.modal-body > form > div.row.glyph-dialog__buttons > div > button.btn.btn-primary.float-end")
        try:
            save.click()
            save.click()
            save.click()
        except:
            pass
        driver.implicitly_wait(10)

    download = driver.find_element_by_css_selector("#toolbar > div.btn-toolbar.ms-auto > div:nth-child(6) > button:nth-child(1)")
    download.click()

    driver.quit()