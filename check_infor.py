import json
from datetime import datetime



def check_information(data_cv, data_uv, user_id):
    name = ''
    phone = ''
    email = ''
    sex = ''
    birthday = ''
    age = ''
    result = {}
    # đọc cv
    name_cv = data_cv['name']
    print('name:', name_cv)
    position_cv = data_cv['position']
    birthday_cv = data_cv['menu'][0]['content']['content']['content']['birthday']
    sex_cv = data_cv['menu'][0]['content']['content']['content']['sex']
    phone_cv = data_cv['menu'][0]['content']['content']['content']['phone']
    email_cv = data_cv['menu'][0]['content']['content']['content']['email']
    address_cv = data_cv['menu'][0]['content']['content']['content']['address']
    if sex_cv.lower() == 'nam' or sex_cv.lower() == 'male':
        sex_cv = '1'
    elif sex_cv.lower() == 'nữ' or sex_cv.lower() == 'female':
        sex_cv = '2'
    # if '年' in birthday_cv:
    #     birthday_cv = birthday_cv.replace('年', '/')
    #     birthday_cv = birthday_cv.replace('月', '/')
    #     birthday_cv = birthday_cv.replace('日', '/')
    #     birthday_cv = datetime.strptime(birthday_cv, "%Y/%m/%d")
    #     birthday_cv = int(birthday_cv.timestamp())
    # elif birthday_cv != '':
    #     birthday_cv = datetime.strptime(birthday_cv, "%d/%m/%Y")
    #     birthday_cv = int(birthday_cv.timestamp())
    # else:
        birthday_cv = ''

    # đọc uv
    use_id = data_uv['use_id']
    use_name = data_uv['use_name']
    use_phone = data_uv['use_phone']
    use_mail = data_uv['use_mail']
    use_city = data_uv['use_city']
    use_gioi_tinh = data_uv['use_gioi_tinh']
    use_birth_day = data_uv['use_birth_day']
    cv_address = data_uv['cv_address']
    cv_cate_id = data_uv['cv_cate_id']
    if name_cv != use_name:
        name = name_cv
    if phone_cv != use_phone:
        phone = phone_cv
    if email_cv != use_mail:
        email = email_cv
    if birthday_cv != use_birth_day:
        birthday = birthday_cv
    if sex_cv != use_gioi_tinh:
        sex = sex_cv
    if address_cv != cv_address:
        address = address_cv
    result[user_id] = {}
    result[user_id]['name'] = name
    result[user_id]['age'] = age
    result[user_id]['sex'] = sex
    result[user_id]['phone'] = phone
    result[user_id]['email'] = email
    result[user_id]['birthday'] = birthday
    return result


def information_check(infor, data_uv, user_id):
    name = ''
    phone = ''
    email = ''
    sex = ''
    birthday = ''
    address = ''
    age = ''
    result = {}

    name_cv = infor['name']
    phone_cv = infor['phone']
    email_cv = infor['email']
    gender_cv = infor['gender']
    birthday_cv = infor['birthday']
    age = infor['age']

    if gender_cv.lower() == 'nam' or gender_cv.lower() == 'male':
        gender_cv = '1'
    elif gender_cv.lower() == 'nữ' or gender_cv.lower() == 'female':
        gender_cv = '2'
    # if birthday_cv != '':
    #     birthday_cv = datetime.strptime(birthday_cv, "%d/%m/%Y")
    #     birthday_cv = int(birthday_cv.timestamp())
    print(111)

    # đọc uv
    use_id = data_uv['use_id']
    use_name = data_uv['use_name']
    use_phone = data_uv['use_phone']
    use_mail = data_uv['use_mail']
    use_gioi_tinh = data_uv['use_gioi_tinh']
    use_birth_day = data_uv['use_birth_day']
    use_address = data_uv['cv_address']
    print(222)
    if name_cv != use_name:
        name = name_cv
    if phone_cv != use_phone:
        phone = phone_cv
    if email_cv != use_mail:
        email = email_cv
    if birthday_cv != use_birth_day:
        birthday = birthday_cv
    if gender_cv != use_gioi_tinh:
        sex = gender_cv
    result[user_id] = {}
    result[user_id]['name'] = name
    result[user_id]['age'] = age
    result[user_id]['sex'] = sex
    result[user_id]['phone'] = phone
    result[user_id]['email'] = email
    result[user_id]['birthday'] = birthday
    print(result)
    return result
