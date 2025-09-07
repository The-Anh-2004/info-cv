import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from OCR_server.crop_box_img import YOLO_Detect

# Map nhãn
LABEL_MAP = {0: 'avatar', 1: 'block', 2: 'infor', 3: 'job_title', 4: 'name'}

def pdf2image(pdf_path, dpi=200):
    """
    Đọc file PDF thành 1 ảnh (ghép dọc tất cả trang).
    """
    pages = convert_from_path(pdf_path, dpi)
    image = np.vstack([np.asarray(page) for page in pages])
    return image

def draw_yolo_boxes(image, yolo_detector, out_path="output_detect.jpg", label_map=None, conf_thres=0.4):
    """
    Vẽ bounding box YOLO và xuất ra file ảnh.
    image: numpy.ndarray (ảnh BGR)
    yolo_detector: instance YOLO_Detect đã load weight
    out_path: đường dẫn lưu ảnh kết quả
    label_map: dict {class_id: class_name}
    conf_thres: chỉ vẽ box có độ tự tin > ngưỡng này
    """
    boxes, labels, detect_image, confs = yolo_detector(image)
    img_show = image.copy()
    if label_map is None:
        label_map = LABEL_MAP
    for box, label, conf in zip(boxes, labels, confs):
        x1, y1, x2, y2 = map(int, box)
        class_name = label_map.get(int(label), str(label))
        if conf < conf_thres:
            continue
        # Vẽ hình chữ nhật
        cv2.rectangle(img_show, (x1, y1), (x2, y2), (0,255,0), 2)
        # Ghi tên class + độ tự tin
        cv2.putText(
            img_show, f"{class_name} {conf:.2f}",
            (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2, cv2.LINE_AA
        )
    cv2.imwrite(out_path, img_show)
    print(f"Saved detection image to: {out_path}")
    return img_show

if __name__ == "__main__":
    # --- Load model YOLO ---
    yolo_detector = YOLO_Detect(weight_path='weights/best.pt')

    # --- Đường dẫn input ---
    input_path = "cv_1752210951522.pdf"   # Có thể là pdf hoặc ảnh

    # --- Đọc ảnh hoặc PDF ---
    ext = os.path.splitext(input_path)[-1].lower()
    if ext == '.pdf':
        img = pdf2image(input_path, dpi=200)
        # pdf2image trả về ảnh RGB, chuyển về BGR cho OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    # --- Vẽ box, xuất ảnh ---
    out_img = draw_yolo_boxes(img, yolo_detector, out_path="output_cv_detect.jpg")
    # Nếu muốn hiển thị luôn (mở ảnh lên xem)
    # cv2.imshow("YOLO detect", out_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

