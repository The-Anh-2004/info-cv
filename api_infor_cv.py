from flask import Flask, request
import urllib.request
from urllib.request import Request
import json
# from check_infor import check_information, information_check
from extract_infor import *
from extract_job import extract_jobtitle
from convert import pdf2image, docx2pdf
from extract_address import extract_city, extract_district, extract_ward, extract_city_and_text_before, extract_city_and_sentence, extract_city_and_text
import os
import cv2
import uuid
import traceback
from OCR_server.crop_box_img import YOLO_Detect

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

file_log = open("ner_err.txt", "a")

app = Flask(__name__)

@app.route('/get_infor', methods=['POST', 'GET'])
def get_infor():
    # Prepare folder for data if not exists
    folder_image = 'folder_data'
    if not os.path.exists(folder_image):
        os.mkdir(folder_image)

    data = None
    error = None

    # Validate input parameters
    user_id = None
    link_image = None
    try:
        user_id = request.form['user_id']
    except KeyError:
        # If not in form, try args or values
        user_id = request.values.get('user_id')
    try:
        link_image = request.values['link_image']
    except KeyError:
        link_image = request.form.get('link_image')
    if not user_id or not link_image:
        # Missing required parameter
        message = "Thiếu thông tin 'user_id' hoặc 'link_image'"  # "Missing user_id or link_image"
        error = ErrorModel(400, message)
        response = ResponseModel(None, vars(error))
        return json.dumps(vars(response))

    # Determine file extension
    extension = os.path.splitext(link_image)[1].lower()
    supported_pdf = ['.pdf']
    supported_doc = ['.docx', '.doc']
    supported_img = ['.jpg', '.jpeg', '.png']

    try:
        # Validate file type
        if extension not in supported_pdf + supported_doc + supported_img:
            # Unsupported file format
            message = "Yêu cầu cung cấp link file PDF, Word hoặc ảnh hợp lệ"  # "Please provide a valid PDF, Word document, or image link."
            error = ErrorModel(400, message)
            # Construct response with error
            response = ResponseModel(None, vars(error))
            return json.dumps(vars(response))

        # Download the file to a local path
        unique_name = str(uuid.uuid4())
        path_local = os.path.join(folder_image, unique_name + extension)
        try:
            req = urllib.request.Request(link_image, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(path_local, 'wb') as out_file:
                out_file.write(resp.read())
        except Exception as download_err:
            # Failed to download the file from the provided URL
            message = f"Không thể tải tệp từ đường dẫn: {download_err}"  # "Cannot download file from the URL"
            error = ErrorModel(400, message)
            response = ResponseModel(None, vars(error))
            return json.dumps(vars(response))

        # Initialize YOLO detection model (consider loading once globally for performance)
        yolo_det_weight = 'weights/best.pt'
        det_model = YOLO_Detect(weight_path=yolo_det_weight)

        # Process based on file type
        if extension in supported_pdf:
            # Convert PDF to images (get list of pages as images)
            image_list = pdf2image(path_local, dpi=200)
            # Take the first page for processing
            if isinstance(image_list, list):
                image = image_list[0]
            else:
                # If pdf2image returns a numpy array stack
                image = image_list[0] if image_list.ndim == 4 else image_list
            # If image is very tall, crop height to 2000 for easier processing
            if image.shape[0] > 2000:
                image = image[0:2000, 0:image.shape[1]]
            # Extract text regions using YOLO and OCR
            name_list, info_list = extract_text(image, det_model)
            text = ' '.join(info_list)
            city_id, city_name,text_no_city_sentences = extract_city_and_sentence(text, city_json_path='api-base365.City.json')
            district_id, district_name,text_no_district = extract_district(text_no_city_sentences, city_id, district_json_path='api-base365.District.json')
            ward_id, ward_name = extract_ward(text_no_district,city_id, ward_json_path='api-base365.Ward.json')
            title_cv_list = extract_job_title(image, det_model)
            title_cv = title_cv_list[0] if title_cv_list else ''
            # Determine job title using extracted text and category JSON
            cat_id, cat_name = extract_jobtitle(text, cat_json_path='api-base365.CategoryJob.json')
            title_list = [cat_name] if cat_name else []  # ensure title is a list
            # Compile all information into the output structure
            infors = extract_infor_img(name_list, info_list, cat_id, cat_name, user_id, job_title=title_cv,
                                      city_id=city_id, city_name=city_name,
                                      district_id=district_id, district_name=district_name,
                                      ward_id=ward_id, ward_name=ward_name)

        elif extension in supported_doc:
            # Convert Word document to PDF
            try:
                # Use docx2pdf convert function: output PDF to same folder
                docx2pdf(path_local, folder_image)
            except Exception as conv_err:
                message = f"Lỗi khi chuyển đổi Word sang PDF: {conv_err}"  # "Error converting Word to PDF"
                error = ErrorModel(500, message)
                response = ResponseModel(None, vars(error))
                return json.dumps(vars(response))
            # Construct the expected PDF path (same name with .pdf extension)
            pdf_path = os.path.splitext(path_local)[0] + '.pdf'
            if not os.path.exists(pdf_path):
                message = "Chuyển đổi sang PDF thất bại, không tìm thấy tệp PDF."
                error = ErrorModel(500, message)
                response = ResponseModel(None, vars(error))
                return json.dumps(vars(response))
            # Convert the first page of PDF to image
            image_list = pdf2image(pdf_path, dpi=200)
            if isinstance(image_list, list):
                image = image_list[0]
            else:
                image = image_list[0] if image_list.ndim == 4 else image_list
            if image.shape[0] > 2000:
                image = image[0:2000, 0:image.shape[1]]
            # YOLO detection and OCR
            name_list, info_list = extract_text(image, det_model)
            text = ' '.join(info_list)
            city_id, city_name,text_no_city_sentences = extract_city_and_text_before(text, city_json_path='api-base365.City.json')
            district_id, district_name,text_no_district = extract_district(text_no_city_sentences, city_id, district_json_path='api-base365.District.json')
            ward_id, ward_name = extract_ward(text_no_district, city_id, ward_json_path='api-base365.Ward.json')
            title_cv_list = extract_job_title(image, det_model)
            title_cv = title_cv_list[0] if title_cv_list else ''
            # Determine job title (using the text info and category JSON)
            cat_id, cat_name = extract_jobtitle(text, cat_json_path='api-base365.CategoryJob.json')
            infors = extract_infor_img(name_list, info_list, cat_id, cat_name, user_id, job_title=title_cv,
                                      city_id=city_id, city_name=city_name,
                                      district_id=district_id, district_name=district_name,
                                      ward_id=ward_id, ward_name=ward_name)

        elif extension in supported_img:
            # Read image using OpenCV
            image = cv2.imread(path_local)
            if image is None:
                message = "Không thể đọc ảnh từ tệp đã tải xuống."
                error = ErrorModel(400, message)
                response = ResponseModel(None, vars(error))
                return json.dumps(vars(response))
            if image.shape[0] > 2000:
                image = image[0:2000, 0:image.shape[1]]
            # YOLO detection and OCR
            name_list, info_list = extract_text(image, det_model)
            text = ' '.join(info_list)
            cat_id, cat_name = extract_jobtitle(text, cat_json_path='api-base365.CategoryJob.json')
            city_id, city_name,text_no_city_sentences = extract_city_and_text_before(text, city_json_path='api-base365.City.json')
            district_id, district_name,text_no_district = extract_district(text_no_city_sentences, city_id, district_json_path='api-base365.District.json')
            ward_id, ward_name = extract_ward(text_no_district, city_id, ward_json_path='api-base365.Ward.json')
            title_cv_list = extract_job_title(image, det_model)
            title_cv = title_cv_list[0] if title_cv_list else ''
            infors = extract_infor_img(name_list, info_list, cat_id, cat_name, user_id, job_title=title_cv,
                                      city_id=city_id, city_name=city_name,
                                      district_id=district_id, district_name=district_name,
                                      ward_id=ward_id, ward_name=ward_name)

        else:
            # This branch theoretically won't be reached due to earlier extension check
            infors = {}
            message = "Định dạng tệp không được hỗ trợ."

        # If we reach here, assume success
        message = "Lấy thông tin thành công"
        data_model = DataModel(True, message, infors)
        data = vars(data_model)

    except Exception as err:
        # Catch-all for unexpected errors
        traceback.print_exc()
        message = f"Lỗi xử lý: {err}"  # "Processing error"
        error = ErrorModel(500, message)
        data = None

    # Prepare the final ResponseModel
    response = ResponseModel(data, vars(error) if error else None)
    return json.dumps(vars(response))
'''def get_infor():
    data_body = dict(request.form)
    error = None
    data = None
    if not os.path.exists('folder_data'):
        os.mkdir('folder_data')
    folder_image = 'folder_data'
    try:
        user_id = data_body['user_id']
        link_image = request.values['link_image']
        extension = os.path.splitext(link_image)[1]
        ext_pdf = ['.pdf']
        ext_doc = ['.docx', '.doc']
        ext_img = ['.jpg', 'jpeg', '.png']
        yolo_det_weight = 'weights/best.pt'
        det_model = YOLO_Detect(weight_path=yolo_det_weight)
        try:
            path_img = folder_image + '/' + str(uuid.uuid4()) + extension
            # Sử dụng Request để thêm User-Agent header
            req = urllib.request.Request(link_image, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(path_img, 'wb') as out_file:
                out_file.write(response.read())
            # path_img giờ đã có ảnh được tải về
        except Exception as err:
            print('err:', err)
        
        if extension in ext_pdf:
            image_list = pdf2image(path_img, 200)  # Giả sử trả về list ảnh hoặc mảng nhiều ảnh
            if isinstance(image_list, list):
                image = image_list[0]
            else:
                # Nếu là numpy array nhiều trang (N, H, W, C), lấy trang đầu
                image = image_list[0] if image_list.ndim == 4 else image_list
            print('size:', image.shape)
            if image.shape[0] > 2000:
                image = image[0:int(2000), 0:image.shape[1], :]
            name, infor = extract_text(image, det_model)
            cat_id, cat_name = extract_jobtitle(infor,cat_json_path='api-base365.CategoryJob.json')
            infors = extract_infor_img(name, infor, title, user_id)
        elif extension in ext_doc:
            docx2pdf(path_img, 'folder_data')
            image_list = pdf2image(path_img, 200)
            if isinstance(image_list, list):
                image = image_list[0]
            else:
                image = image_list[0] if image_list.ndim == 4 else image_list
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
        traceback.print_exc()
        print('err:', err)
        error = ErrorModel(200, message)
    if data is not None:
        data = vars(data)
    if error is not None:
        error = vars(error)
    response = ResponseModel(data, error)
    return json.dumps(vars(response))
'''

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8082)  
