import cv2
import torch
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

import OCR_server.craft_utils as craft_utils
import OCR_server.img_proc as imgproc
from OCR_server.craft import CRAFT
from OCR_server.refinenet import RefineNet
from collections import OrderedDict

# --- copyStateDict như các file trước ---
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

# --- Dùng lại test_net để lấy box dòng ---
def test_net(
    net, image, text_threshold, link_threshold, low_text, cuda, poly, args, refine_net=None
):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = torch.unsqueeze(x, 0).float()
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    net = net.to(device)
    if refine_net is not None:
        refine_net = refine_net.to(device)
    net.eval()
    with torch.no_grad():
        y, feature = net(x)
    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().numpy()
    boxes, polys, det_scores = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]
    return boxes, polys, det_scores

# --- Hàm crop theo polygon (cho dòng) ---
def crop_poly(img, poly):
    rect = cv2.boundingRect(np.array(poly, np.int32))
    x, y, w, h = rect
    crop_img = img[y:y+h, x:x+w]
    # Mask để loại bỏ vùng ngoài polygon nếu cần
    poly = np.array(poly, np.int32)
    poly = poly - [x, y]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    crop_img = cv2.bitwise_and(crop_img, crop_img, mask=mask)
    # Đổi nền ngoài polygon thành trắng (tùy chọn)
    white_bg = np.ones_like(crop_img, np.uint8) * 255
    crop_img = np.where(mask[...,None]==255, crop_img, white_bg)
    return crop_img

# --- Định nghĩa args giả lập cho test_net ---
class Args:
    def __init__(self):
        self.canvas_size = 1280
        self.mag_ratio = 1.8
        self.show_time = False

if __name__ == "__main__":
    # 1. Load mô hình VietOCR
    config = Cfg.load_config_from_file('OCR_server/configOCR/vgg-transformer.yml')
    config['weights'] = 'weights/transformerocr.pth'
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    detector = Predictor(config)

    # 2. Load mô hình CRAFT + RefineNet để lấy dòng
    craft = CRAFT()
    craft.load_state_dict(copyStateDict(torch.load('weights/craft_mlt_25k.pth', map_location='cpu')))
    craft.eval()
    refine_net = RefineNet()
    refine_net.load_state_dict(copyStateDict(torch.load('weights/craft_refiner_CTW1500.pth', map_location='cpu')))
    refine_net.eval()
    args = Args()

    # 3. Đọc ảnh block (ảnh crop vùng 'infor' hoặc bất kỳ)
    img = cv2.imread('craft_detect_result.jpg')
    if img is None:
        raise FileNotFoundError("Không tìm thấy file cv_infor_crop.jpg")

    # 4. Detect dòng bằng CRAFT
    boxes, polys, scores = test_net(
        craft, img, 
        text_threshold=0.6, 
        link_threshold=0.45, 
        low_text=0.32, 
        cuda=False, 
        poly=True, 
        args=args, 
        refine_net=refine_net
    )

    # 5. Crop từng dòng và nhận diện bằng VietOCR
    print("Kết quả nhận diện text từng dòng:")
    for i, poly in enumerate(polys):
        if poly is not None:
            crop_img = crop_poly(img, poly)
            # Chuyển crop_img về PIL cho VietOCR
            crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            pred = detector.predict(crop_pil)
            print(f"Line {i+1}: {pred}")

            # Tùy chọn: lưu lại từng dòng để debug
            #cv2.imwrite(f"line_{i+1}.jpg", crop_img)

    print("Xong!")

