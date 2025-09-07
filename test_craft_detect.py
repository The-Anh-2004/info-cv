import cv2
import os
import torch
import numpy as np
from collections import OrderedDict
from OCR_server.craft import CRAFT
from OCR_server.refinenet import RefineNet
import OCR_server.craft_utils as craft_utils
import OCR_server.img_proc as imgproc
from pdf2image import convert_from_path
from OCR_server.test_craft import test_net

# --- Hàm copyStateDict ---
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

def pdf2image_cv(pdf_path, dpi=200):
    """
    Đọc file PDF thành 1 ảnh (ghép dọc tất cả trang), trả về ảnh RGB.
    """
    pages = convert_from_path(pdf_path, dpi)
    image = np.vstack([np.asarray(page) for page in pages])
    return image

class Args:
    def __init__(self):
        self.canvas_size = 1280
        self.mag_ratio = 1.8
        self.show_time = True

if __name__ == "__main__":
    input_path = "cv_1751185211929.pdf"   # Có thể là pdf hoặc ảnh

    ext = os.path.splitext(input_path)[-1].lower()
    if ext == '.pdf':
        img = pdf2image_cv(input_path, dpi=200)
        # convert_from_path trả ảnh RGB, cần chuyển sang BGR cho OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    craft_weight_path = "weights/craft_mlt_25k.pth"
    refiner_weight_path = "weights/craft_refiner_CTW1500.pth"

    device = 'cpu'  # Đổi thành 'cuda' nếu có GPU

    # 1. Load model CRAFT
    craft = CRAFT()
    craft.load_state_dict(copyStateDict(torch.load(craft_weight_path, map_location=device)))
    craft = craft.to(device)
    craft.eval()

    # 2. Load Refiner
    refine_net = RefineNet()
    refine_net.load_state_dict(copyStateDict(torch.load(refiner_weight_path, map_location=device)))
    refine_net = refine_net.to(device)
    refine_net.eval()

    # 3. Tạo args cho test_net
    args = Args()

    # 4. Chạy CRAFT detect (ảnh đầu vào đã chuẩn)
    boxes, polys, heatmap, det_scores = test_net(
        net=craft,
        image=img,
        text_threshold=0.6,
        link_threshold=0.45,
        low_text=0.32,
        cuda=(device=='cuda'),
        poly=True,
        args=args,
        refine_net=refine_net
    )

    # 5. Vẽ kết quả lên ảnh
    img_vis = img.copy()
    for box in boxes:
        if box is not None:
            pts = np.array(box).astype(np.int32)
            cv2.polylines(img_vis, [pts], isClosed=True, color=(0,0,255), thickness=2)
    cv2.imwrite("craft_detect_result.jpg", img_vis)
    print("Đã lưu kết quả detect: craft_detect_result.jpg")
