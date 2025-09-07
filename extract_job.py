'''"import re
import json
import os
from collections import Counter, defaultdict

TITLE_KEYWORDS = [
    r'TRƯỞNG BỘ PHẬN [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'GIÁM ĐỐC [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'PHÓ [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'CHUYÊN VIÊN [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'NHÂN VIÊN [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'TRƯỞNG PHÒNG [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'PHÓ PHÒNG [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+'
]

def make_cat_keywords(cat_name):
    """Sinh các từ khóa đặc trưng cho từng ngành nghề"""
    base = [w.strip() for w in re.split(r'[-,]', cat_name) if w.strip()]
    # Bổ sung rule riêng cho từng ngành nếu muốn:
    custom = {
    "Kế toán - Kiểm toán": ["kế toán", "kiểm toán", "accounting", "audit", "financial report", "balance sheet", "financial analysis"],
    "Hành chính - Văn phòng": ["hành chính", "văn phòng", "office", "administration", "lễ tân", "văn thư"],
    "Việc làm thời vụ": ["thời vụ", "part time", "seasonal"],
    "Sinh viên làm thêm": ["sinh viên", "ngoài giờ", "student", "part-time internship"],
    "Xây dựng": ["xây dựng", "construction", "công trình"],
    "Điện - Điện tử": ["điện", "điện tử", "electrical", "electronics", "electronic"],
    "Làm bán thời gian": ["bán thời gian", "part time"],
    "Vận tải - Lái xe": ["lái xe", "vận tải", "driver", "transportation"],
    "Khách sạn - Nhà hàng": ["khách sạn", "nhà hàng", "hotel", "restaurant"],
    "Nhân viên kinh doanh": ["kinh doanh", "sales", "business development"],
    "Việc làm bán hàng": ["bán hàng", "sale", "sell", "sales"],
    "Cơ khí - Chế tạo": ["cơ khí", "chế tạo", "mechanical", "machining"],
    "Lao động phổ thông": ["lao động phổ thông", "manual labor"],
    "IT phần mềm": ["it", "phần mềm", "lập trình", "software", "developer"],
    "Marketing - PR": ["marketing", "pr", "quảng cáo", "truyền thông"],
    "Nhập liệu": ["nhập liệu", "data entry"],
    "Giáo dục - Đào tạo": ["giáo dục", "đào tạo", "teaching", "training"],
    "Kỹ thuật": ["kỹ thuật", "engineering", "technician"],
    "Y tế - Dược": ["y tế", "dược", "healthcare", "medical", "pharmacy"],
    "Quản trị kinh doanh": ["quản trị kinh doanh", "business administration", "mba"],
    "Dịch vụ": ["dịch vụ", "service", "support"],
    "Biên - Phiên dịch": ["biên dịch", "phiên dịch", "translate", "interpreter"],
    "Dệt may - Da giày": ["dệt may", "da giày", "textile", "garment", "shoe"],
    "Xuất - nhập khẩu": ["xuất nhập khẩu", "export", "import", "logistic", "shipping"],
    "IT Phần cứng - mạng": ["phần cứng", "mạng", "hardware", "network", "it infrastructure"],
    "Nhân sự": ["nhân sự", "hr", "human resources", "recruitment"],
    "Thiết kế - Mỹ thuật": ["thiết kế", "mỹ thuật", "design", "designer", "graphic"],
    "Tư vấn": ["tư vấn", "consultant", "advisory"],
    "Bảo vệ": ["bảo vệ", "security"],
    "Ô tô - xe máy": ["ô tô", "xe máy", "auto", "motorbike", "car"],
    "Thư ký - Trợ lý": ["thư ký", "trợ lý", "assistant", "secretary"],
    "KD bất động sản": ["bất động sản", "real estate"],
    "Du lịch": ["du lịch", "travel", "tour", "tourism"],
    "Báo chí - Truyền hình": ["báo chí", "truyền hình", "journalism", "media"],
    "Thực phẩm - Đồ uống": ["thực phẩm", "đồ uống", "food", "beverage", "fnb"],
    "Ngành nghề khác": [],
    "Vật tư - Thiết bị": ["vật tư", "thiết bị", "equipment", "supplies"],
    "Thiết kế web": ["web", "thiết kế web", "web design", "frontend", "backend"],
    "In ấn - Xuất bản": ["in ấn", "xuất bản", "printing", "publishing"],
    "Nông - Lâm - Ngư - Nghiệp": ["nông", "lâm", "ngư nghiệp", "agriculture", "farming"],
    "Thương mại điện tử": ["thương mại điện tử", "ecommerce", "e-commerce", "online shopping"],
    "Việc làm thêm tại nhà": ["làm tại nhà", "work from home", "remote"],
    "Chăm sóc khách hàng": ["chăm sóc khách hàng", "customer service", "cs"],
    "Sinh viên mới tốt nghiệp - Thực tập": ["thực tập", "intern", "mới tốt nghiệp"],
    "Kỹ thuật ứng dụng": ["kỹ thuật ứng dụng", "applied engineering"],
    "Bưu chính viễn thông": ["viễn thông", "bưu chính", "telecom", "postal"],
    "Dầu khí - Địa chất": ["dầu khí", "địa chất", "oil", "gas", "petroleum"],
    "Giao thông vận tải - Thủy lợi - Cầu đường": ["giao thông", "thủy lợi", "cầu đường", "transport infrastructure"],
    "Khu chế xuất - Khu công nghiệp": ["khu công nghiệp", "khu chế xuất", "industrial park"],
    "Làm đẹp - Thể lực - Spa": ["spa", "làm đẹp", "fitness", "gym"],
    "Luật - Pháp lý": ["luật", "pháp lý", "legal", "law"],
    "Môi trường - Xử lý chất thải": ["môi trường", "xử lý chất thải", "environment", "waste"],
    "Mỹ phẩm - Thời trang - Trang sức": ["mỹ phẩm", "thời trang", "trang sức", "cosmetic", "fashion", "beauty"],
    "Ngân hàng - Chứng khoán - Đầu tư": ["ngân hàng", "chứng khoán", "đầu tư", "bank", "finance", "investment"],
    "Nghệ thuật - Điện ảnh": ["nghệ thuật", "điện ảnh", "art", "film"],
    "Phát triển thị trường": ["phát triển thị trường", "market development", "bd"],
    "Phục vụ - Tạp vụ": ["phục vụ", "tạp vụ", "service", "janitor"],
    "Quan hệ đối ngoại": ["quan hệ đối ngoại", "ngoại giao", "public relations", "PR"],
    "Quản lý điều hành": ["quản lý", "điều hành", "management", "admin"],
    "Sản xuất - Vận hành sản xuất": ["sản xuất", "vận hành", "manufacturing", "production"],
    "Thẩm định - Giám thẩm định - Quản lý chất lượng": ["thẩm định", "giám định", "chất lượng", "qc", "inspection"],
    "Thể dục - Thể thao": ["thể dục", "thể thao", "fitness", "sports"],
    "Hóa học - Sinh học": ["hóa học", "sinh học", "chemistry", "biology"],
    "Bảo hiểm": ["bảo hiểm", "insurance"],
    "Freelancer": ["freelancer", "freelance", "tự do"],
    "Công chức - Viên chức": ["công chức", "viên chức", "public servant"],
    "Điện tử viễn thông": ["điện tử", "viễn thông", "telecommunication", "iot"],
    "Hoạch định - Dự án": ["dự án", "project", "planning", "hoạch định"],
    "Lương cao": ["lương cao", "high salary", "well-paid"],
    "Tiếp thị - Quảng cáo": ["tiếp thị", "quảng cáo", "marketing"],
    "Việc làm Tết": ["việc làm tết", "tet job"],
    "Giúp việc": ["giúp việc", "housekeeping", "helper"],
    "Thủy sản": ["thủy sản", "fisheries", "aquaculture"],
    "Công nghệ thực phẩm": ["thực phẩm", "food technology", "food science"],
    "Chăn nuôi - Thú y": ["chăn nuôi", "thú y", "animal care"],
    "An toàn lao động": ["an toàn lao động", "safety", "osha"],
    "Hàng không": ["hàng không", "aviation"],
    "Tài chính": ["tài chính", "finance"],
    "Tổ chức sự kiện": ["sự kiện", "event", "event planning"],
    "Trắc địa": ["trắc địa", "surveying", "land survey"],
    "Bảo trì": ["bảo trì", "maintenance"],
    "Hàng hải": ["hàng hải", "maritime", "shipping"],
    "Đầu bếp - phụ bếp": ["đầu bếp", "bếp", "chef", "cook", "kitchen"],
    "Truyền thông": ["truyền thông", "media", "communications"],
    "Startup": ["startup", "khởi nghiệp"],
    "Thư viện": ["thư viện", "library"],
    "Thống kê": ["thống kê", "statistics"],
    "Copywriter": ["copywriter", "content writer", "viết nội dung"],
    "Xuất khẩu lao động": ["xuất khẩu lao động", "overseas worker"],
    "Công nghệ cao": ["công nghệ cao", "high tech"],
    "Pha chế - Bar": ["pha chế", "bar", "bartender"],
    "Lễ tân - PG - PB": ["lễ tân", "pg", "pb", "receptionist"],
    "Logistic": ["logistic", "logistics", "giao nhận"],
    "Vận chuyển giao nhận": ["giao nhận", "forwarding"],
    "Quản lý đơn hàng": ["đơn hàng", "order management"],
    "Thu ngân": ["thu ngân", "cashier"],
    "Telesales": ["telesales", "call center", "tele sales"]
}
    return [*base, *custom.get(cat_name, [])]

def extract_jobtitle(text, cat_json_path="api-base365.CategoryJob.json"):
    text_upper = text.upper()

    # 1. Trích xuất job title bằng regex
    job_titles = []
    for pattern in TITLE_KEYWORDS:
        matches = re.findall(pattern, text_upper)
        for m in matches:
            title = m.strip().replace('I ', '').replace('|', '')
            if title and title not in job_titles:
                job_titles.append(title)
    main_job_title = job_titles[0] if job_titles else None

    # 2. Load danh mục
    if not os.path.exists(cat_json_path):
        raise FileNotFoundError(f"Không tìm thấy file category: {cat_json_path}")
    with open(cat_json_path, "r", encoding="utf-8") as f:
        cats = json.load(f)

    # 3. So khớp job title trước (nếu có)
    if main_job_title:
        best_cat = None
        best_score = 0
        for cat in cats:
            cat_name = cat["cat_name"]
            keywords = make_cat_keywords(cat_name)
            score = sum(1 for word in keywords if word.upper() in main_job_title)
            if score > best_score:
                best_cat = cat
                best_score = score
        if best_cat and best_score > 0:
            return best_cat["cat_id"], best_cat["cat_name"]

    # 4. Nếu không, so khớp từng từ khóa với toàn bộ text (ưu tiên ngành có nhiều từ khóa trùng nhất)
    best_cat = None
    best_score = 0
    for cat in cats:
        cat_name = cat["cat_name"]
        keywords = make_cat_keywords(cat_name)
        score = sum(text_upper.count(word.upper()) for word in keywords)
        if score > best_score:
            best_cat = cat
            best_score = score
    if best_cat and best_score > 0:
        return best_cat["cat_id"], best_cat["cat_name"]

    # 5. Nếu không có ngành nào, trả về "Ngành nghề khác"
    for cat in cats:
        if "NGÀNH NGHỀ KHÁC" in cat["cat_name"].upper():
            return cat["cat_id"], cat["cat_name"]
    return None, None'''

import re
import json
import os
from flashtext import KeywordProcessor
from collections import Counter, defaultdict

# --- Bước 0: Build các từ khóa ---
def build_keyword_processor(custom):
    kp = KeywordProcessor(case_sensitive=False)
    for cat_name, words in custom.items():
        for w in words:
            kp.add_keyword(w, cat_name)
    return kp

def make_custom_keywords():
    return {
    "Kế toán - Kiểm toán": ["kế toán", "kiểm toán", "accounting", "audit", "financial report", "balance sheet", "financial analysis","financial statement", "financial management"],
    "Hành chính - Văn phòng": ["hành chính", "văn phòng", "office", "administration", "lễ tân", "văn thư","secretary", "administrative assistant", "office manager"],
    "Việc làm thời vụ": ["thời vụ", "part time", "seasonal", "temporary job", "seasonal work"],
    "Sinh viên làm thêm": ["sinh viên", "ngoài giờ", "student", "part-time internship", "student job", "student internship"],
    "Xây dựng": ["xây dựng", "construction", "công trình", "building", "civil engineering", "site manager"],
    "Điện - Điện tử": ["điện", "điện tử", "electrical", "electronics", "electronic", "electrical engineering", "electronics technician"],
    "Làm bán thời gian": ["bán thời gian", "part time", "part-time", "flexible hours"],
    "Vận tải - Lái xe": ["lái xe", "vận tải", "driver", "transportation", "logistics", "delivery", "chauffeur", "truck driver"],
    "Khách sạn - Nhà hàng": ["khách sạn", "nhà hàng", "hotel", "restaurant", "hospitality", "food service", "waiter", "waitress", "cook", "chef"],
    "Nhân viên kinh doanh": ["kinh doanh", "sales", "business development", "business development executive", "sales representative", "account manager"],
    "Việc làm bán hàng": ["bán hàng", "sale", "sell", "sales", "retail", "sales associate", "sales clerk", "retail sales"],
    "Cơ khí - Chế tạo": ["cơ khí", "chế tạo", "mechanical", "machining", "mechanical engineering", "machinist", "fabrication"],
    "Lao động phổ thông": ["lao động phổ thông", "manual labor", "general labor", "unskilled labor", "laborer"],
    "IT phần mềm": ["it", "phần mềm", "lập trình", "software", "developer","typeScript", "tavaScript","frontend" ,"frameworks","backend","java","css"],
    "Marketing - PR": ["marketing", "pr", "quảng cáo", "truyền thông", "advertising", "public relations", "marketing specialist", "digital marketing", "social media"],
    "Nhập liệu": ["nhập liệu", "data entry", "data input", "data processing", "data entry clerk"],
    "Giáo dục - Đào tạo": ["giáo dục", "đào tạo", "teaching", "training", "education", "instructor", "teacher", "trainer", "lecturer"],
    "Kỹ thuật": ["kỹ thuật", "engineering", "technician", "technical", "engineering technician", "field engineer", "maintenance technician"],
    "Y tế - Dược": ["y tế", "dược", "healthcare", "medical", "pharmacy", "nursing", "doctor", "nurse", "pharmacist", "healthcare assistant"],
    "Quản trị kinh doanh": ["quản trị kinh doanh", "business administration", "mba", "business management", "business analyst", "management"],
    "Dịch vụ": ["dịch vụ", "service", "support", "customer service", "service representative", "service technician"],
    "Biên - Phiên dịch": ["biên dịch", "phiên dịch", "translate", "interpreter", "translation", "interpretation", "translator", "interpreter"],
    "Dệt may - Da giày": ["dệt may", "da giày", "textile", "garment", "shoe", "textile engineering", "garment manufacturing", "shoe manufacturing"],
    "Xuất - nhập khẩu": ["xuất nhập khẩu", "export", "import", "logistic", "shipping", "logistics", "supply chain", "export manager", "import manager"],
    "IT Phần cứng - mạng": ["phần cứng", "mạng", "hardware", "network", "it infrastructure", "iot", "internet of things", "cloud", "server", "networking"],
    "Nhân sự": ["nhân sự", "hr", "human resources", "recruitment", "talent acquisition", "personnel"],
    "Thiết kế - Mỹ thuật": ["thiết kế", "mỹ thuật", "design", "designer", "graphic", "art", "creative", "graphic design", "web design", "ux/ui design"],
    "Tư vấn": ["tư vấn", "consultant", "advisory", "consulting", "business consultant", "management consultant", "advisory services"],
    "Bảo vệ": ["bảo vệ", "security"],
    "Ô tô - xe máy": ["ô tô", "xe máy", "auto", "motorbike", "car", "automotive", "vehicle", "car mechanic", "motorbike technician"],
    "Thư ký - Trợ lý": ["thư ký", "trợ lý", "assistant", "secretary", "administrative assistant", "executive assistant", "personal assistant"],
    "KD bất động sản": ["bất động sản", "real estate", "property", "real estate agent", "real estate broker", "property manager"],
    "Du lịch": ["du lịch", "travel", "tour", "tourism","travel agent", "tour guide", "tour operator", "hospitality management"],
    "Báo chí - Truyền hình": ["báo chí", "truyền hình", "journalism", "media", "journalist", "reporter", "broadcasting", "media production"],
    "Thực phẩm - Đồ uống": ["thực phẩm", "đồ uống", "food", "beverage", "fnb", "food service", "food production", "beverage production", "food and beverage manager"],
    "Ngành nghề khác": [],
    "Vật tư - Thiết bị": ["vật tư", "thiết bị", "equipment", "supplies", "materials", "inventory", "supply chain management"],
    "Thiết kế web": ["web", "thiết kế web", "web design", "frontend", "backend", "full stack", "web developer", "web designer", "html", "css", "javascript"],
    "In ấn - Xuất bản": ["in ấn", "xuất bản", "printing", "publishing", "print", "publisher", "printing press", "graphic arts"],
    "Nông - Lâm - Ngư - Nghiệp": ["nông", "lâm", "ngư nghiệp", "agriculture", "farming", "forestry", "fishing", "aquaculture"],
    "Thương mại điện tử": ["thương mại điện tử", "ecommerce", "e-commerce", "online shopping", "digital commerce", "e-commerce specialist", "online sales"],
    "Việc làm thêm tại nhà": ["làm tại nhà", "work from home", "remote", "home-based", "telecommute", "remote work", "freelance"],
    "Chăm sóc khách hàng": ["chăm sóc khách hàng", "customer service", "cs", "customer support", "client service", "customer care", "call center"],
    "Sinh viên mới tốt nghiệp - Thực tập": ["thực tập", "intern", "mới tốt nghiệp", "graduate", "entry-level", "internship", "trainee", "new graduate"],
    "Kỹ thuật ứng dụng": ["kỹ thuật ứng dụng", "applied engineering", "applied technology", "application engineering", "applied technician"],
    "Bưu chính viễn thông": ["viễn thông", "bưu chính", "telecom", "postal"],
    "Dầu khí - Địa chất": ["dầu khí", "địa chất", "oil", "gas", "petroleum"],
    "Giao thông vận tải - Thủy lợi - Cầu đường": ["giao thông", "thủy lợi", "cầu đường", "transport infrastructure"],
    "Khu chế xuất - Khu công nghiệp": ["khu công nghiệp", "khu chế xuất", "industrial park"],
    "Làm đẹp - Thể lực - Spa": ["spa", "làm đẹp", "fitness", "gym"],
    "Luật - Pháp lý": ["luật", "pháp lý", "legal", "law"],
    "Môi trường - Xử lý chất thải": ["môi trường", "xử lý chất thải", "environment", "waste"],
    "Mỹ phẩm - Thời trang - Trang sức": ["mỹ phẩm", "thời trang", "trang sức", "cosmetic", "fashion", "beauty"],
    "Ngân hàng - Chứng khoán - Đầu tư": ["ngân hàng", "chứng khoán", "đầu tư", "bank", "finance", "investment"],
    "Nghệ thuật - Điện ảnh": ["nghệ thuật", "điện ảnh", "art", "film"],
    "Phát triển thị trường": ["phát triển thị trường", "market development", "bd"],
    "Phục vụ - Tạp vụ": ["phục vụ", "tạp vụ", "service", "janitor"],
    "Quan hệ đối ngoại": ["quan hệ đối ngoại", "ngoại giao", "public relations", "PR"],
    "Quản lý điều hành": ["quản lý", "điều hành", "management", "admin"],
    "Sản xuất - Vận hành sản xuất": ["sản xuất", "vận hành", "manufacturing", "production"],
    "Thẩm định - Giám thẩm định - Quản lý chất lượng": ["thẩm định", "giám định", "chất lượng", "qc", "inspection"],
    "Thể dục - Thể thao": ["thể dục", "thể thao", "fitness", "sports"],
    "Hóa học - Sinh học": ["hóa học", "sinh học", "chemistry", "biology"],
    "Bảo hiểm": ["bảo hiểm", "insurance"],
    "Freelancer": ["freelancer", "freelance", "tự do", "independent contractor", "self-employed"],
    "Công chức - Viên chức": ["công chức", "viên chức", "public servant"],
    "Điện tử viễn thông": ["điện tử", "viễn thông", "telecommunication", "iot", "internet of things", "telecom engineer"],
    "Hoạch định - Dự án": ["dự án", "project", "planning", "hoạch định", "project management", "project planner", "project coordinator"],
    "Lương cao": ["lương cao", "high salary", "well-paid", "high income", "lucrative"],
    "Tiếp thị - Quảng cáo": ["tiếp thị", "quảng cáo", "marketing", "advertising", "marketing specialist", "advertising executive"],
    "Việc làm Tết": ["việc làm tết", "tet job", "tet employment", "tết"],
    "Giúp việc": ["giúp việc", "housekeeping", "helper", "domestic worker", "housekeeper"],
    "Thủy sản": ["thủy sản", "fisheries", "aquaculture"],
    "Công nghệ thực phẩm": ["thực phẩm", "food technology", "food science"],
    "Chăn nuôi - Thú y": ["chăn nuôi", "thú y", "animal care"],
    "An toàn lao động": ["an toàn lao động", "safety", "osha"],
    "Hàng không": ["hàng không", "aviation", "airline", "flight attendant", "pilot", "air traffic control"],
    "Tài chính": ["tài chính", "finance", "financial services", "financial analyst", "investment banking"],
    "Tổ chức sự kiện": ["sự kiện", "event", "event planning", "hoạch định sự kiện", "event coordinator", "event manager"],
    "Trắc địa": ["trắc địa", "surveying", "land survey"],
    "Bảo trì": ["bảo trì", "maintenance", "repair", "maintenance technician", "service technician"],
    "Hàng hải": ["hàng hải", "maritime", "shipping"],
    "Đầu bếp - phụ bếp": ["đầu bếp", "bếp", "chef", "cook", "kitchen"],
    "Truyền thông": ["truyền thông", "media", "communications", "media relations", "communications specialist", "media planner"],
    "Startup": ["startup", "khởi nghiệp"],
    "Thư viện": ["thư viện", "library"],
    "Thống kê": ["thống kê", "statistics", "data analysis", "statistician"],
    "Copywriter": ["copywriter", "content writer", "viết nội dung", "content creation", "copywriting"],
    "Xuất khẩu lao động": ["xuất khẩu lao động", "overseas worker", "expat", "foreign worker"],
    "Công nghệ cao": ["công nghệ cao", "high tech", "high technology", "advanced technology", "tech"],
    "Pha chế - Bar": ["pha chế", "bar", "bartender", "mixologist", "barista"],
    "Lễ tân - PG - PB": ["lễ tân", "pg", "pb", "receptionist"],
    "Logistic": ["logistic", "logistics", "giao nhận", "logistics coordinator", "logistics manager"],
    "Vận chuyển giao nhận": ["giao nhận", "forwarding"],
    "Quản lý đơn hàng": ["đơn hàng", "order management"],
    "Thu ngân": ["thu ngân", "cashier", "cash handling", "teller"],
    "Telesales": ["telesales", "call center", "tele sales"]
}

CUSTOM = make_custom_keywords()
kp = build_keyword_processor(CUSTOM)

# --- Patterns regex để trích job title ---
TITLE_KEYWORDS = [
    r'TRƯỞNG BỘ PHẬN [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'GIÁM ĐỐC [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'PHÓ [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'CHUYÊN VIÊN [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'NHÂN VIÊN [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'TRƯỞNG PHÒNG [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+',
    r'PHÓ PHÒNG [A-ZĂÂÁÀẠẢÃÂẤẦẬẨẪẮẰẶẲẴEÊÉÈẸẺẼẾỀỆỂỄIÍÌỊỈĨOÔƠÓÒỌỎÕỐỒỘỔỖỚỜỢỞỠUƯÚÙỤỦŨỨỪỰỬỮYÝỲỴỶỸ ]+'
]
COMPILED_REGEX = [re.compile(p) for p in TITLE_KEYWORDS]

def extract_jobtitle(text, cat_json_path="api-base365.CategoryJob.json"):
    text_upper = text.upper()
    # 1. Trích job title bằng regex
    main_job_title = None
    for pat in COMPILED_REGEX:
        m = pat.search(text_upper)
        if m:
            main_job_title = m.group().strip()
            break

    # 2. Load danh mục
    if not os.path.exists(cat_json_path):
        raise FileNotFoundError(cat_json_path)
    with open(cat_json_path, encoding="utf-8") as f:
        cats = json.load(f)

    # 3. Nếu có job title, ưu tiên dùng FlashText để phân loại
    if main_job_title:
        matches = kp.extract_keywords(main_job_title)
        if matches:
            # lấy cat_name xuất hiện nhiều nhất
            best = Counter(matches).most_common(1)[0][0]
            for cat in cats:
                if cat["cat_name"] == best:
                    return cat["cat_id"], cat["cat_name"]

    # 4. Nếu không có job title hoặc không match, dùng FlashText cho toàn text
    matches = kp.extract_keywords(text)
    if matches:
        best = Counter(matches).most_common(1)[0][0]
        for cat in cats:
            if cat["cat_name"] == best:
                return cat["cat_id"], cat["cat_name"]

    # 5. Không match => Ngành nghề khác
    for cat in cats:
        if "NGÀNH NGHỀ KHÁC" in cat["cat_name"].upper():
            return cat["cat_id"], cat["cat_name"]
    return None, None
