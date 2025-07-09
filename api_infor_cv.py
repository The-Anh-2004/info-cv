from flask import Flask, request
import urllib.request
from urllib.request import Request
import json
# from check_infor import check_information, information_check
from extract_infor import *
from convert import pdf2image, docx2pdf
import os
import cv2
import uuid
import sys

# sys.path.insert(0, '/home/hungha/AI_365/timviec365_elasticsearch/infor_cv')

class ErrorModel:
    def __init__(self, code, message):
        self.code = code
        self.message = message

class ResponseModel:
    # Trả về phản hồi gồm data và error
    def __init__(self, data, error):
        self.data = data
        self.error = error

class DataModel:
    def __init__(self, result, message, item):
        self.result = result
        self.message = message
        self.item = item

file_log = open("/home/hungha/AI_365copy/timviec365_elasticsearch/infor_uv/ner_err.txt", "a")

app = Flask(__name__)

@app.route('/get_infor', methods=['POST', 'GET'])
def get_infor():
    data_body = dict(request.form)
    error = None
    data = None
    if not os.path.exists('/home/hungha/AI_365copy/timviec365_elasticsearch/infor_uv/folder_data'):
        os.mkdir('/home/hungha/AI_365copy/timviec365_elasticsearch/infor_uv/folder_data')
    folder_image = '/home/hungha/AI_365copy/timviec365_elasticsearch/infor_uv/folder_data'
    try:
        user_id = data_body['user_id']
        link_image = request.values['link_image']
        extension = os.path.splitext(link_image)[1]
        ext_pdf = ['.pdf']
        ext_doc = ['.docx', '.doc']
        ext_img = ['.jpg', 'jpeg', '.png']

        yolo_det_weight = '/home/hungha/AI_365copy/timviec365_elasticsearch/infor_uv/model/best.pt'
        det_model = YOLO_Detect(weight_path=yolo_det_weight)
        try:
            opener = urllib.request.URLopener()
            opener.addheader('User-Agent', 'whatever')
            path_img = folder_image + '/' + str(uuid.uuid4()) + extension
            filename, headers = opener.retrieve(link_image, path_img)

            # path_img = urllib.request.urlretrieve(link_image, folder_image + '/' + str(uuid.uuid4()) + extension)
        except Exception as err:
            print('err:', err)
        
        if extension in ext_pdf:
            image = pdf2image(path_img, 200)
            # image = pdf2image(path_img[0], 200)
            print('size:', image.shape)
            if image.shape[0] > 2000:
                image = image[0:int(2000), 0:image.shape[1], :]
            name, infor = extract_text(image, det_model)
            title = extract_jobtitle(image, det_model)
            infors = extract_infor_img(name, infor, title, user_id)
        elif extension in ext_doc:
            docx2pdf(path_img, 'folder_data')
            image = pdf2image(path_img, 200)
            if image.shape[0] > 2000:
                image = image[0:int(2000), 0:image.shape[1], :]
            name, infor = extract_text(image, det_model)
            title = extract_jobtitle(image, det_model)
            infors = extract_infor_img(name, infor, title, user_id)
        elif extension in ext_img:
            image = cv2.imread(path_img)
            if image.shape[0] > 2000:
                image = image[0:int(2000), 0:image.shape[1], :]
            name, infor = extract_text(image, det_model)
            title = extract_jobtitle(image, det_model)
            print('name:', name)
            print('infor:', infor)
            infors = extract_infor_img(name, infor, title, user_id)
        else:
            message = 'yêu cầu truyền đúng định dạng link'
            infors = {}
            infors['email'] = ''
            infors['phone'] = ''
            infors['birthday'] = ''
            infors['age'] = ''
            infors['gender'] = ''
            infors['name'] = ''
            infors['title_cv'] = ''
            infors['address'] = ''
            infors['user_id'] = ''
        message = 'Lấy thông tin thành công'
        print('infors:', infors)
        data = DataModel(True, message, infors)
    except Exception as err:
        message = 'Thông tin truyền lên không đầy đủ'
        print('err:', err)
        error = ErrorModel(200, message)
    if data is not None:
        data = vars(data)
    if error is not None:
        error = vars(error)
    response = ResponseModel(data, error)
    return json.dumps(vars(response))


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8082)
