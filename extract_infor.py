# với cv ứng viên tải lên, input là link, chỉ đọc tên, ngành nghề, số điện thoại, email, địa chỉ, ngày sinh, giới tính.
from ultralytics import YOLO
import torch
import cv2
import os
from get_address import extract_address
import easyocr
import fitz
import re
import docx
# from underthesea import ner
import spacy
### OCR ###
import torch
import cv2
import os
import matplotlib.pyplot as plt
# import shutil
from OCR_server.testCraft import *
from OCR_server.craft import *
from OCR_server.pipeline import *
from OCR_server.inference import *
from OCR_server.load_model import *
# import glob
# import numpy as np
# from pdf2image import convert_from_path
# from flask import Flask,request
# import json
# from PIL import Image
# import uuid
# import io
# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
# import argparse
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# from torch.autograd import Variable
# import urllib
from unidecode import unidecode
# import multiprocessing

def chuyen_cau_khong_dau_chu_thuong(cau):
    # Chuyển đổi văn bản có dấu thành văn bản không dấu
    cau_khong_dau = unidecode(cau)
    
    # Chuyển câu thành chữ thường
    cau_khong_dau_chu_thuong = cau_khong_dau.lower()
    
    return cau_khong_dau_chu_thuong
def process_cropped_image(cropped_img):
    list_crop_line = crop_image_line(cropped_img,craft,args,refine_net)
    return list_crop_line
    
def extractText(image): 
    list_crop_line = crop_image_line_info(image,craft,args,refine_net)

    _ouput, all_info_box = recog(list_crop_line,ocr_model_1)
    text = _ouput["title"] + ' ' +  _ouput["text"]
    return text



det_box_model, ocr_model, craft, refine_net, args, ocr_model_1 = get_model()


### OCR ###

map_label = {0: 'avatar',
             1: 'block',
             2: 'infor',
             3: 'job_title',
             4: 'name'}

class YOLO_Detect:
    def __init__(self, weight_path=None, model_name=None):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weight_path)
        self.model.to(self.device)

    def __call__(self, image, return_result=False, output_path=None):
        result = self.model(image, conf=0.25, verbose=False, nms=True, iou=0.5)
        boxes_list = result[0].boxes.data[:, :4]
        label_idx = result[0].boxes.data[:, 5]
        #labels = [map_label[int(i)] for i in label_idx]
        box = result[0].boxes
        conf = box.conf

        detect_image = result[0].plot()
        return boxes_list, label_idx, detect_image, conf
    

def extract_text(image, det_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    output_path = '/home/hungha/AI_365copy/timviec365_elasticsearch/infor_uv/results'
    det_model = det_model
    boxes_list, label_list, detect_image, confs = det_model(image, output_path= output_path, return_result=True)
    name = []
    infor = []
    for i in range(len(boxes_list)):
        if label_list[i] == torch.tensor(4.) :
            cropped_img = image[int(boxes_list[i][1]):int(boxes_list[i][3]), int(boxes_list[i][0]):int(boxes_list[i][2]), :]
            cv2.imwrite(os.path.join('/home/hungha/AI_365copy/timviec365_elasticsearch/infor_uv/folder_data/' + str(i) + '.jpg'), cropped_img)
            text = extractText(cropped_img)
            name.append(text)
        if label_list[i] == torch.tensor(2.):
            cropped_img = image[int(boxes_list[i][1]):int(boxes_list[i][3]), int(boxes_list[i][0]):int(boxes_list[i][2]), :]
            cv2.imwrite(os.path.join('/home/hungha/AI_365copy/timviec365_elasticsearch/infor_uv/folder_data/' + str(i) + '.jpg'), cropped_img)
            text = extractText(cropped_img)
            infor.append(text)

    return name, infor

def extract_jobtitle(image, det_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    output_path = './results'
    det_model = det_model
    boxes_list, label_list, detect_image_job, confs = det_model(image, output_path= output_path, return_result=True)
    cv2.imwrite(os.path.join('/home/hungha/AI_365copy/timviec365_elasticsearch/job_cv/folder_data/detect_image.jpg'), detect_image_job)
    job_title = []
    for i in range(len(boxes_list)):
        if label_list[i] == torch.tensor(3.) :
            cropped_img = image[int(boxes_list[i][1]):int(boxes_list[i][3]), int(boxes_list[i][0]):int(boxes_list[i][2]), :]
            cv2.imwrite(os.path.join('./folder_data/' + str(i) + '.jpg'), cropped_img)
            text = extractText(cropped_img)
            job_title.append(text)

    return job_title

def pdf2text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text+=page.get_text()
    return text

def doc2text(path):
    doc = docx.Document(path)
    '''
    except:
        if doc is None:
            convert_to_docx(filename)
            file = filename.replace('.doc', '.docx')
            os.remove(filename)
            doc = docx.Document(file)
    '''
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    infor =  '\n'.join(fullText)
    return infor



def check_text(text1, text2):
    count = 0
    for t1 in text1:
        if t1 in text2:
            count = count + 1
    rate = 2*count/(len(text1)+len(text2))
    return rate


def extract_gender(text):
    text = text.replace('Việt Nam', '')
    text = text.replace('việt nam', '')
    text = text.replace('Vietnam', '')
    text = text.replace('vietnam', '')
    patterns = ['Nam', 'Nữ', 'Male', 'Female', 'nam', 'nữ', 'NAM', 'NỮ']
    count = 0
    for pattern in patterns:
        if pattern in text:
            gender = pattern
            count = count + 1
            break
    if count == 0:
        gender = 'Khác'
    return gender


def extract_email(text):
    patterns = [r'[\w\.-]+@[\w\.-]+']
    text = text.lower()
    text = text.replace(' ogmail', '@gmail')
    text = text.replace('ogmail', '@gmail')
    text = text.replace('ogmail.com', '@gmail.com')
    text = text.replace('qgmail', '@gmail')
    text = text.replace('0gmail com', '@gmail.com')
    text = text.replace('0gmail.com', '@gmail.com')
    text = text.replace('@gmail.Com', '@gmail.com')
    text = text.replace('gmailcom', 'gmail.com')
    for pattern in patterns:
        if re.findall(pattern, text):
            email = re.findall(pattern, text)
            return email[0]
    return None

def extract_phone(text):
    patterns = ['[0-9]{10}', '[0-9]{5} [0-9]{5}',
                '[0-9]{4} [0-9]{3} [0-9]{3}',
                '[0-9]{4}.[0-9]{3}.[0-9]{3}',
                '[0-9]{3} [0-9]{3} [0-9]{4}',
                '[0-9]{3}.[0-9]{3}.[0-9]{4}',
                '[0-9]{4}-[0-9]{3}-[0-9]{3}',
                '[0-9]{3}-[0-9]{3}-[0-9]{4}']
    for pattern in patterns:
        if (re.findall(pattern, text)):
            return re.findall(pattern, text)[0]
    return None
    

def extract_date_of_birth(text):
    text = text.replace('年 ', '年')
    text = text.replace(' 年', '年')
    text = text.replace('月 ', '月')
    text = text.replace(' 年', '年')
    text = text.replace('日 ', '日')
    text = text.replace(' 日', '日')
    patterns = ['[A-Za-z]+\s\d{1,2},\s\d{4}', '[A-Za-z]+\s\d{1,2}(?:st|nd|rd|th)?\s\d{4}', '\s\d{1,2} [A-Za-z]+ \s\d{4}',
                '\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}', 
                '[0-9]{4}年[0-9]{2}月[0-9]{2}日', '[A-Za-z]+\s\d{1,2}(?:St|Nd|Rd|Th)?\s\d{4}', '\d{2}\s[/.-]\s\d{2}\s[/.-]\s\d{4}',
                '\d{1}[/]\d{2}[/]\d{4}', '\d{1}[/]\d{1}[/]\d{4}', '\d{1}[-]\d{2}[-]\d{4}', '\d{1}[.]\d{1}[.]\d{4}', '\d{1}[-]\d{1}[-]\d{4}',
                '\d{2}[.]\d{1}[.]\d{4}', '\d{2}[-]\d{1}[-]\d{4}',  '\d{1}[.]\d{2}[.]\d{4}', '\d{2}[-]\d{2}[-]\d{4}',  '\d{4}[-]\d{2}[-]\d{2}',
                '\d{2}[/]\d{2}[/]\d{4}', '\d{4}[/]\d{2}[/]\d{2}', '\d{4}[.]\d{2}[.]\d{2}', '\d{2}[/]\d{1}[/]\d{4}',  '\d{2}[.]\d{2}[.]\d{4}']
    birth = ''
    age = ''
    for pattern in patterns:
        if re.findall(pattern, text):
            for date in re.findall(pattern, text):
                if re.findall('[0-9]{4}', date) and int(re.findall('[0-9]{4}', date)[0]) < 2006:
                    birth = date
                    age = 2023 - int(re.findall('[0-9]{4}', birth)[0])
                    break
    return birth, age

def show_text(arr):
    if len(arr) > 0:
        text = arr[0]
        for i in range(1, len(arr)):
            text = text+', '+arr[i]
    else:
        text = ''
    return text

# def extract_address(text):
#     address = []
#     tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base", model_max_length=50)
#     try:
#         model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
#         print(1)
#         nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
#     except Exception as err:
#         print('err1:', err)
#     print(2)
#     ner_results = nlp(text)
#     for ent in ner_results:
#         if (ent['entity_group'] == 'LOCATION'):
#             address.append(ent['word'])
#     return show_text(address)

# def extract_address(text):
#     address = []
#     for add in ner(text):
#         if 'LOC' in add[-1]:
#             address.append(add[0])
#     address = ' '.join([str(add) for add in address])
#     return address


def extract_infor_img(name, infor, title, user_id):
    phone = ''
    email = ''
    birthday = ''
    gender = ''
    address = ''
    age = ''
    fullname = ''
    infors = {}
    for text in infor:
        print('text:', text)
        if extraxt_email(text) != None:
            email = extraxt_email(text)
        if extract_phone(text) != None:
            phone = extract_phone(text)
    for text in infor:
        birthday, age = extract_date_of_birth(text)
        if birthday != '':
            break
    for text in infor:
        gender = extract_gender(text)
        if gender != 'Khác':
            break
    for text in infor:
        address = extract_address(text)
        address = address.replace(' /', '/')
        address = address.replace('/ ', '/')
        address = address.replace(' / ', '/')
        address = address.replace('. ', '.')
        if address != '':
            break
    if len(name) > 0:
        fullname = name[0]
    else:
        fullname = ''
    if len(title) > 0:
        title_cv = title[0]
    else:
        title_cv = ''
    infors['email'] = email
    infors['phone'] = phone
    infors['birthday'] = birthday
    infors['gender'] = gender
    infors['age'] = age
    infors['name'] = fullname
    infors['title_cv'] = title_cv
    infors['address'] = address
    infors['user_id'] = user_id
    return infors

def extract_infor(text, user_id):
    phone = ''
    email = ''
    birthday = ''
    gender = ''
    title_cv = ''
    fullname = ''
    address = ''
    age = ''
    infors = {}
    if extraxt_email(text) != None:
        email = extraxt_email(text)
    if extract_phone(text) != None:
        phone = extract_phone(text)
    gender = extract_gender(text)
    birthday, age = extract_date_of_birth(text)
    nlp_ner = spacy.load("/home/hungha/AI_365copy/timviec365_elasticsearch/infor_uv/model")
    doc = nlp_ner(text)
    for entity in doc.ents:
        if entity.label_ == 'PERSON':
            fullname = entity.text
            break
    address = extract_address(text)
    infors['email'] = email
    infors['phone'] = phone
    infors['birthday'] = birthday
    infors['gender'] = ''
    infors['title_cv'] = title_cv
    infors['name'] = fullname
    infors['age'] = age
    infors['address'] = address
    infors['user_id'] = user_id
    return infors


