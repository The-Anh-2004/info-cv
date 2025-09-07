import json
import os,re
from flashtext import KeywordProcessor
from unidecode import unidecode
import unicodedata

# Danh sách các viết tắt phổ biến cho một số thành phố lớn và các tỉnh
CITY_ABBREVIATIONS = {
    "An Giang": ["an giang", "angiang"],
    "Bà Rịa Vũng Tàu": [
        "vung tau", "ba ria vung tau", "baria vungtau", "vungtau", "bariavungtau", "baria-vungtau", "baria -vungtau", "baria- vungtau"
    ],
    "Bắc Giang": ["bac giang", "bacgiang"],
    "Bắc Kạn": ["bac kan", "backan"],
    "Bạc Liêu": ["bac lieu", "baclieu"],
    "Bắc Ninh": ["bac ninh", "bacninh"],
    "Bến Tre": ["ben tre", "bentre"],
    "Bình Định": ["binh dinh", "binhdinh"],
    "Bình Dương": ["binh duong", "binhduong"],
    "Bình Phước": ["binh phuoc", "binhphuoc"],
    "Bình Thuận": ["binh thuan", "binhthuan"],
    "Cà Mau": ["ca mau", "camau"],
    "Cần Thơ": ["can tho", "cantho"],
    "Cao Bằng": ["cao bang", "caobang"],
    "Đà Nẵng": ["da nang", "danang"],
    "Đắk Lắk": ["dak lak", "daklak"],
    "Đắk Nông": ["dak nong", "daknong"],
    "Điện Biên": ["dien bien", "dienbien"],
    "Đồng Nai": ["dong nai", "dongnai"],
    "Đồng Tháp": ["dong thap", "dongthap"],
    "Gia Lai": ["gia lai", "gialai"],
    "Hà Giang": ["ha giang", "hagiang"],
    "Hà Nam": ["ha nam", "hanam"],
    "Hà Nội": ["ha noi", "hanoi","hn"],
    "Hà Tĩnh": ["ha tinh", "hatinh"],
    "Hải Dương": ["hai duong", "haiduong"],
    "Hải Phòng": ["hai phong", "haiphong"],
    "Hậu Giang": ["hau giang", "haugiang"],
    "Hòa Bình": ["hoa binh", "hoabinh"],
    "Hưng Yên": ["hung yen", "hungyen"],
    "Khánh Hòa": ["khanh hoa", "khanhhoa"],
    "Kiên Giang": ["kien giang", "kiengiang"],
    "Kon Tum": ["kon tum", "kontum"],
    "Lai Châu": ["lai chau", "laichau"],
    "Lâm Đồng": ["lam dong", "lamdong"],
    "Lạng Sơn": ["lang son", "langson"],
    "Lào Cai": ["lao cai", "laocai"],
    "Long An": ["long an", "longan"],
    "Nam Định": ["nam dinh", "namdinh", "nam đinh"],
    "Nghệ An": ["nghe an", "nghean"],
    "Ninh Bình": ["ninh binh", "ninhbinh"],
    "Ninh Thuận": ["ninh thuan", "ninhthuan"],
    "Phú Thọ": ["phu tho", "phutho"],
    "Phú Yên": ["phu yen", "phuyen"],
    "Quảng Bình": ["quang binh", "quangbinh"],
    "Quảng Nam": ["quang nam", "quangnam"],
    "Quảng Ngãi": ["quang ngai", "quangngai"],
    "Quảng Ninh": ["quang ninh", "quangninh"],
    "Quảng Trị": ["quang tri", "quangtri"],
    "Sóc Trăng": ["soc trang", "soctrang"],
    "Sơn La": ["son la", "sonla"],
    "Tây Ninh": ["tay ninh", "tayninh"],
    "Thái Bình": ["thai binh", "thaibinh"],
    "Thái Nguyên": ["thai nguyen", "thainguyen"],
    "Thanh Hóa": ["thanh hoa", "thanhhoa"],
    "Thừa Thiên Huế": ["thua thien hue", "thuathienhue"],
    "Tiền Giang": ["tien giang", "tiengiang"],
    "Trà Vinh": ["tra vinh", "travinh"],
    "Tuyên Quang": ["tuyen quang", "tuyenquang"],
    "Vĩnh Long": ["vinh long", "vinhlong"],
    "Vĩnh Phúc": ["vinh phuc", "vinhphuc"],
    "Yên Bái": ["yen bai", "yenbai"],
    "Hồ Chí Minh": ["hcm", "tp.hcm", "tphcm", "ho chi minh", "hochiminh"]
}

def split_sentences(text):
    # Chia câu dựa trên các dấu . ! ?
    sentences = re.split(r'(?<=[.!?])+', text)
    return sentences

def remove_single_keyword_from_text(text, keyword):
    """
    Xóa đúng 1 lần đầu tiên keyword (có thể là biến thể: không dấu, viết liền, hoa thường...) khỏi text.
    """
    # Sinh các biến thể thông dụng của keyword
    variants = [
        keyword,
        keyword.lower(),
        keyword.upper(),
        keyword.title(),
        unidecode(keyword),
        unidecode(keyword).lower(),
        unidecode(keyword).upper(),
        unidecode(keyword).title(),
        keyword.replace(" ", ""),
        unidecode(keyword.replace(" ", "")),
    ]
    variants = list(dict.fromkeys(variants))
    variants.sort(key=lambda x: -len(x))
    
    # Chuẩn hóa chuỗi đầu vào thành không dấu để tìm kiếm
    text_norm = unidecode(text)
    for variant in variants:
        # Chuẩn hóa variant thành không dấu
        variant_norm = unidecode(variant)
        # Tìm vị trí xuất hiện (không phân biệt hoa thường) trong chuỗi không dấu
        idx = text_norm.lower().find(variant_norm.lower())
        if idx != -1:
            # Kiểm tra ranh giới từ (trước/sau không phải là ký tự chữ, số, _)
            prev_char = text_norm[idx-1] if idx > 0 else ""
            next_idx = idx + len(variant_norm)
            next_char = text_norm[next_idx] if next_idx < len(text_norm) else ""
            if (not prev_char.isalnum() and prev_char != "_") and (not next_char.isalnum() and next_char != "_"):
                # Xác định đoạn tương ứng trên chuỗi gốc và xóa nó
                text = text[:idx] + " " + text[next_idx:]
                break
    # Làm sạch chuỗi sau khi xóa
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*,", ",", text)
    text = text.replace(" ,", ",").replace(", ", ", ")
    text = re.sub(r"(\s+,|,\s+)", ", ", text)
    return text.strip()

def remove_all_variants_from_text(text, keyword):
    """
    Xoá toàn bộ các biến thể có thể của keyword khỏi text.
    Dùng để xử lý xong district thì loại triệt để, tránh lặp lại ở ward.
    """
    variants = [
        keyword,
        keyword.lower(),
        keyword.upper(),
        keyword.title(),
        unidecode(keyword),
        unidecode(keyword).lower(),
        unidecode(keyword).upper(),
        unidecode(keyword).title(),
        keyword.replace(" ", ""),
        unidecode(keyword.replace(" ", "")),
    ]
    variants = list(dict.fromkeys(variants))
    # Sắp xếp theo độ dài giảm dần
    variants.sort(key=lambda x: -len(x))
    for variant in variants:
        # Tìm tất cả và xoá hết
        pattern = re.compile(r'(?<!\w)'+re.escape(variant)+r'(?!\w)', re.IGNORECASE)
        text = pattern.sub(" ", text)
    # Clean lại text
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*,", ",", text)
    text = text.replace(" ,", ",").replace(", ", ", ")
    text = re.sub(r"(\s+,|,\s+)", ", ", text)
    return text.strip()

def make_city_keywords(city_json_path="api-base365.City.json"):
    """Sinh ra dict từ khóa cho từng tỉnh/thành, hỗ trợ các biến thể phổ biến."""
    if not os.path.exists(city_json_path):
        raise FileNotFoundError(city_json_path)
    with open(city_json_path, encoding="utf-8") as f:
        cities = json.load(f)

    custom = {}
    for city in cities:
        name = city["name"]
        name_nodau = unidecode(name)
        keywords = set([
            name,
            name.lower(),
            name.title(),
            name.upper(),
            name_nodau,
            name_nodau.lower(),
            name_nodau.title(),
            name_nodau.upper()
        ])
        # Thêm các biến thể/viết tắt nếu có
        if name in CITY_ABBREVIATIONS:
            for abbr in CITY_ABBREVIATIONS[name]:
                keywords.add(abbr)
                keywords.add(abbr.lower())
                keywords.add(unidecode(abbr.lower()))
        custom[name] = list(keywords)
    return custom

def build_city_keyword_processor(city_json_path="api-base365.City.json"):
    city_keywords = make_city_keywords(city_json_path)
    kp = KeywordProcessor(case_sensitive=False)
    for city_name, variants in city_keywords.items():
        for variant in variants:
            kp.add_keyword(variant.strip(), city_name)
    return kp

def extract_city(text, city_json_path="api-base365.City.json"):
    kp = build_city_keyword_processor(city_json_path)
    matches = kp.extract_keywords(text)
    if not matches:
        return None, None, text

    # Tìm tất cả matches với vị trí trong văn bản
    match_positions = []
    for m in set(matches):  # Loại trùng để tránh check lặp
        idx = text.lower().rfind(m.lower())
        if idx >= 0:
            match_positions.append((idx, m))
    if not match_positions:
        return None, None, text

    # Lấy match xuất hiện sau cùng
    match_positions.sort()
    _, best_city = match_positions[-1]

    with open(city_json_path, encoding="utf-8") as f:
        cities = json.load(f)
    for city in cities:
        if city["name"].lower() == best_city.lower():
            city_id = city["_id"]
            city_name = city["name"]
            text_removed = remove_single_keyword_from_text(text, best_city)
            return city_id, city_name, text_removed

    return None, None, text

#Sau khi đã xác định city, ta chỉ lấy câu có city nhưng cắt bỏ city trong đó
def get_sentence_no_city(text, city_name):
    sentences = split_sentences(text)
    kp = KeywordProcessor(case_sensitive=False)
    # Sinh các biến thể như hàm city keyword
    variants = [
        city_name,
        city_name.lower(),
        city_name.title(),
        city_name.upper(),
        unidecode(city_name),
        unidecode(city_name).lower(),
        unidecode(city_name).title(),
        unidecode(city_name).upper(),
        city_name.replace(" ", ""),
        unidecode(city_name.replace(" ", "")),
    ]
    variants = list(dict.fromkeys(variants))
    for v in variants:
        kp.add_keyword(v.strip(), v.strip())  # Giữ lại đúng variant match

    for s in sentences:
        found = kp.extract_keywords(s)
        if found:
            # Xóa đúng variant đã xuất hiện trong câu!
            s_no_city = remove_single_keyword_from_text(s, found[0])
            return s_no_city.strip()
    return None
def extract_city_and_sentence(text, city_json_path="api-base365.City.json"):
    city_id, city_name, text_removed = extract_city(text, city_json_path)
    if city_name:
        text_no_city_sentence = get_around_keyword_no_city(text_removed, city_name, window=20)
    else:
        text_no_city_sentence = text_removed
    return city_id, city_name, text_no_city_sentence

# Hàm lấy đoạn văn bản xung quanh city keyword, xoá city keyword
def get_around_keyword_no_city(text, city_name, window=20):
    # Sinh các biến thể như hàm city keyword
    variants = [
        city_name,
        city_name.lower(),
        city_name.title(),
        city_name.upper(),
        unidecode(city_name),
        unidecode(city_name).lower(),
        unidecode(city_name).title(),
        unidecode(city_name).upper(),
        city_name.replace(" ", ""),
        unidecode(city_name.replace(" ", "")),
    ]
    variants = list(dict.fromkeys(variants))
    
    # Chia nhỏ text thành các từ
    words = text.split()
    text_join = " ".join(words)  # Đảm bảo đồng bộ
    lower_join = unidecode(text_join.lower())
    
    # Tìm vị trí từng variant trong mảng words
    found_pos = None
    found_variant = None
    for variant in variants:
        v_norm = unidecode(variant.lower())
        # Ghép lại toàn bộ từ thành string, tìm vị trí start của variant
        idx = lower_join.find(v_norm)
        if idx != -1:
            # Xác định vị trí token (word) của keyword bắt đầu tại đâu
            chars = 0
            for i, w in enumerate(words):
                chars += len(w) + 1  # +1 cho khoảng trắng
                if chars > idx:
                    found_pos = i
                    found_variant = variant
                    break
            if found_pos is not None:
                break
    if found_pos is None:
        return None  # Không tìm thấy city
    
    # Cắt đoạn xung quanh, loại keyword ra khỏi đoạn
    left = max(0, found_pos - window//2)
    right = min(len(words), found_pos + window//2)
    segment_words = words[left:right+1]
    # Loại đúng variant khỏi segment (chỉ xóa 1 lần, ưu tiên đúng vị trí)
    segment_str = " ".join(segment_words)
    segment_str = remove_single_keyword_from_text(segment_str, found_variant)
    return segment_str.strip()

def extract_city_and_text(text, city_json_path="api-base365.City.json"):
    city_id, city_name, text_removed = extract_city(text, city_json_path)
    if city_name:
        text_no_city_around = get_around_keyword_no_city(text, city_name, window=20)
    else:
        text_no_city_around = text_removed
    return city_id, city_name, text_no_city_around

# Hàm lấy 20 từ trước city keyword, không lấy city keyword
def get_before_keyword_no_city(text, city_name):
    # Sinh các biến thể như hàm city keyword
    variants = [
        city_name,
        city_name.lower(),
        city_name.title(),
        city_name.upper(),
        unidecode(city_name),
        unidecode(city_name).lower(),
        unidecode(city_name).title(),
        unidecode(city_name).upper(),
        city_name.replace(" ", ""),
        unidecode(city_name.replace(" ", "")),
    ]
    variants = list(dict.fromkeys(variants))
    
    words = text.split()
    text_join = " ".join(words)
    lower_join = unidecode(text_join.lower())
    
    found_pos = None
    found_variant = None
    for variant in variants:
        v_norm = unidecode(variant.lower())
        idx = lower_join.find(v_norm)
        if idx != -1:
            chars = 0
            for i, w in enumerate(words):
                chars += len(w) + 1
                if chars > idx:
                    found_pos = i
                    found_variant = variant
                    break
            if found_pos is not None:
                break
    if found_pos is None:
        return None  # Không tìm thấy city
    
    # Lấy 20 từ trước keyword (nếu thiếu thì lấy hết mức có thể)
    left = max(0, found_pos - 20)
    right = found_pos  # Không lấy keyword, chỉ lấy trước nó
    segment_words = words[left:right]
    segment_str = " ".join(segment_words)
    # Đảm bảo loại bỏ đúng 1 lần keyword nếu nó dính liền phía trước (rare case)
    segment_str = remove_single_keyword_from_text(segment_str, found_variant)
    return segment_str.strip()

def extract_city_and_text_before(text, city_json_path="api-base365.City.json"):
    city_id, city_name, text_removed = extract_city(text, city_json_path)
    if city_name:
        text_no_city_before = get_before_keyword_no_city(text, city_name)
    else:
        text_no_city_before = text_removed
    return city_id, city_name, text_no_city_before


# ---- Hàm sinh từ khóa cho quận/huyện ----
def make_district_keywords(district_json_path, city_id=None):
    if not os.path.exists(district_json_path):
        raise FileNotFoundError(district_json_path)
    with open(district_json_path, encoding="utf-8") as f:
        districts = json.load(f)
    custom = {}
    for district in districts:
        if city_id is not None and district["parent"] != city_id:
            continue
        name = district["name"]
        name_nodau = unidecode(name)
        keywords = set([
            name, name.lower(), name.title(), name.upper(),
            name_nodau, name_nodau.lower(), name_nodau.title(), name_nodau.upper()
        ])
        # Tách bỏ tiền tố như "Huyện", "Quận", "Thành phố", "Thị xã"
        lower_nodau = name_nodau.lower()
        for prefix in ["huyen ", "quan ", "tp ", "thanh pho ", "thi xa ", "thi tran "]:
            if lower_nodau.startswith(prefix):
                pure = lower_nodau[len(prefix):].strip()
                # Thêm biến thể không dấu, viết liền, bỏ cách, viết tắt (q1, h15,...)
                keywords.add(pure)
                keywords.add(pure.replace(" ", ""))
                # Nếu là số, thêm dạng q1, q 1, h15, h 15
                if pure.isdigit():
                    short_prefix = prefix.strip().split(" ")[0][0]  # q, h, t...
                    keywords.add(f"{short_prefix}{pure}")
                    keywords.add(f"{short_prefix} {pure}")
        custom[name] = list(keywords)
    return custom

def build_district_keyword_processor(district_json_path, city_id=None):
    district_keywords = make_district_keywords(district_json_path, city_id)
    kp = KeywordProcessor(case_sensitive=False)
    for district_name, variants in district_keywords.items():
        for variant in variants:
            kp.add_keyword(variant.strip(), district_name)
    return kp

from unidecode import unidecode

def extract_district(text, city_id, district_json_path="api-base365.District.json"):
    kp = build_district_keyword_processor(district_json_path, city_id)
    matches = kp.extract_keywords(text)
    if not matches:
        matches = kp.extract_keywords(unidecode(text))
    if matches:
        best_district = matches[0]
        with open(district_json_path, encoding="utf-8") as f:
            districts = json.load(f)
        for district in districts:
            if district["parent"] == city_id and district["name"] == best_district:
                district_id = district["_id"]
                district_name = district["name"]
                # Loại đúng 1 lần đầu tiên district khỏi text (giống extract_city)
                text_removed = remove_single_keyword_from_text(text, best_district)
                return district_id, district_name, text_removed
    return None, None, text


def make_ward_keywords(ward_json_path, city_id=None):
    if not os.path.exists(ward_json_path):
        raise FileNotFoundError(ward_json_path)
    with open(ward_json_path, encoding="utf-8") as f:
        wards = json.load(f)
    custom = {}
    for ward in wards:
        if city_id is not None and ward["city_id"] != city_id:
            continue  # Chỉ lấy xã/phường thuộc tỉnh/thành đã xác định
        name = ward["name"]
        name_nodau = unidecode(name)
        keywords = set([
            name, name.lower(), name.title(), name.upper(),
            name_nodau, name_nodau.lower(), name_nodau.title(), name_nodau.upper()
        ])
        # Bổ sung biến thể bỏ tiền tố ("phường", "xã", "thị trấn")
        lower_nodau = name_nodau.lower()
        for prefix in ["phuong ", "xa ", "thi tran "]:
            if lower_nodau.startswith(prefix):
                pure = lower_nodau[len(prefix):].strip()
                keywords.add(pure)
                keywords.add(pure.replace(" ", ""))
                # Nếu là số, thêm dạng viết tắt: p10, p 10, x3...
                if pure.isdigit():
                    short_prefix = prefix.strip().split(" ")[0][0]  # p, x, t...
                    keywords.add(f"{short_prefix}{pure}")
                    keywords.add(f"{short_prefix} {pure}")
        custom[name] = list(keywords)
    return custom


def build_ward_keyword_processor(ward_json_path, city_id=None):
    ward_keywords = make_ward_keywords(ward_json_path, city_id)
    kp = KeywordProcessor(case_sensitive=False)
    for ward_name, variants in ward_keywords.items():
        for variant in variants:
            kp.add_keyword(variant.strip(), ward_name)
    return kp

def extract_ward(text, city_id, ward_json_path="api-base365.Ward.json"):
    kp = build_ward_keyword_processor(ward_json_path, city_id)
    matches = kp.extract_keywords(text)
    if not matches:
        matches = kp.extract_keywords(unidecode(text))
    if matches:
        best_ward = matches[0]
        best_ward_nodau = unidecode(best_ward).lower().strip().strip(",. ")
        with open(ward_json_path, encoding="utf-8") as f:
            wards = json.load(f)
        for ward in wards:
            if ward["city_id"] == city_id:
                ward_name_nodau = unidecode(ward["name"]).lower().strip().strip(",. ")
                if best_ward_nodau == ward_name_nodau:
                    return ward["_id"], ward["name"]
    return None, None


# --- DEMO ---
'''if __name__ == "__main__":
    text = (
    "Ứng viên sinh sống tại xã Hợp Đồng, thuộc huyện Chương Mỹ "
    "Trong quá trình làm việc, bạn ấy từng chuyển đến quận Thanh Xuân để tiện cho công việc. "
    "Địa chỉ thường trú là: Số 10, ngõ 123 quận TP. Hà Nội. "
    "Ngoài ra, ứng viên từng có thời gian học tập tại Quận Hoàng Mai - hà nội, cụ thể là tại phường Giáp Bát. "
    "Gần đây nhất, ứng viên vừa mới chuyển về huyện Chương Mỹ, TP HÀ NỘI để sinh sống cùng gia đình."
)

    city_id, city_name, text_no_city_before = extract_city_and_text_before(text, "api-base365.City.json")
    print("Sau khi loại city:", text_no_city_before)
    district_id, district_name, text_no_district = extract_district(text_no_city_before, city_id, "api-base365.District.json")
    print("Sau khi loại district:", text_no_district)
    ward_id, ward_name = extract_ward(text_no_district, city_id, "api-base365.Ward.json")
    print("Tỉnh/thành:", city_id, city_name)
    print("Quận/huyện:", district_id, district_name)
    print("Phường/xã:", ward_id, ward_name)'''
