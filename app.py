# flask_web/app.py

import os

from flask import Flask
from flask import jsonify
from flask import render_template
from flask_bootstrap import Bootstrap

from config import UPLOAD_FOLDER
from utils import FaceNotFoundError
from utils_asr import do_recognize
from utils_face import face_detect, face_verify, face_search
from utils_face_attributes import face_attributes
from utils_match import match_image, match_video
from utils_tag import search_tag

bootstrap = Bootstrap()


def create_app(config_name):
    app = Flask(config_name, static_url_path="", static_folder="static")
    bootstrap.init_app(app)
    from api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api/v1')
    # Bootstrap(app)
    return app


app = create_app(__name__)


@app.route('/detect')
def detect():
    return render_template('face_detect.html')


@app.route('/process_detect', methods=['POST'])
def process_detect():
    num_faces, elapsed, fn, _, _ = face_detect()
    fn = os.path.join(UPLOAD_FOLDER, fn)
    if num_faces > 0:
        result = "图片中已检测到人脸。"
    else:
        result = "图片中没有检测到人类的脸。"
    num_faces = "人脸数量: {}".format(num_faces)
    elapsed = "耗时: {:.4f} 秒".format(elapsed)
    return render_template('result_detect.html', result=result, num_faces=num_faces, fn=fn, elapsed=elapsed)


@app.route('/attributes')
def attributes():
    return render_template('face_attributes.html')


@app.route('/process_attributes', methods=['POST'])
def process_attributes():
    result, elapsed, fn = face_attributes()
    fn = os.path.join(UPLOAD_FOLDER, fn)

    pitch = 0.0
    roll = 0.0
    yaw = 0.0

    if result:
        age = result['age']
        pitch = result['pitch']
        roll = result['roll']
        yaw = result['yaw']
        beauty = result['beauty']
        beauty_prob = result['beauty_prob']
        expression = result['expression']
        gender = result['gender']
        glasses = result['glasses']
        # race = result['race']
        result = '年龄={} 性别={} 颜值={} 表情={} 眼镜={} pitch={} roll={} yaw={}  '.format(age, gender, beauty,
                                                                                  expression, glasses,
                                                                                  pitch, roll, yaw)
        comment = '您的颜值超过了 {0:.2f} % 的人群'.format(beauty_prob * 100)
    else:
        result = '抱歉没有检测到人类的脸。'
        comment = ''

    elapsed = "耗时: {:.4f} 秒".format(elapsed)
    return render_template('result_attributes.html', result=result, comment=comment, pitch=pitch, roll=roll, yaw=yaw,
                           fn=fn, elapsed=elapsed)


@app.route('/', methods=['GET'])
def home():
    return render_template('face_verify.html')


@app.route('/verify', methods=['GET'])
def verify():
    return render_template('face_verify.html')


@app.route('/process_verify', methods=['POST'])
def process_verify():
    try:
        is_same, prob, elapsed, fn_1, fn_2 = face_verify()
        fn_1 = os.path.join(UPLOAD_FOLDER, fn_1)
        fn_2 = os.path.join(UPLOAD_FOLDER, fn_2)
        if is_same:
            result = "验证结果：两张脸属于同一个人。"
        else:
            result = "验证结果：两张脸属于不同的人。"
        prob = "置信度为 {:.5f}".format(prob)
        elapsed = "耗时: {:.4f} 秒".format(elapsed)
    except FaceNotFoundError as err:
        result = '对不起，[{}] 图片中没有检测到人类的脸。'.format(err)
        prob = elapsed = fn_1 = fn_2 = ""

    return render_template('result_verify.html', result=result, fn_1=fn_1, fn_2=fn_2, prob=prob, elapsed=elapsed)


@app.route('/search')
def search():
    return render_template('face_search.html')


@app.route('/process_search', methods=['POST'])
def process_search():
    try:
        name, prob, file_star, file_upload, elapsed = face_search()
        file_star = file_star.replace('data', '')
        file_upload = file_upload.replace('static', '')
        result = '最相似的明星: {}'.format(name)
        prob = "置信度为 {:.5f}".format(prob)
        elapsed = "耗时: {:.4f} 秒".format(elapsed)
    except FaceNotFoundError as err:
        file_upload = str(err)
        file_upload = file_upload.replace('static', '')
        result = '对不起，[{}] 图片中没有检测到人类的脸。'.format(file_upload)
        prob = file_star = elapsed = ""
    return render_template('result_search.html', result=result, prob=prob, file_star=file_star, file_upload=file_upload,
                           elapsed=elapsed)


@app.route('/image-match')
def image_match():
    return render_template('match_image.html')


@app.route('/process_image_match', methods=['POST'])
def process_image_match():
    is_match, elapsed, fn_1, fn_2 = match_image()
    fn_1 = os.path.join(UPLOAD_FOLDER, fn_1)
    fn_2 = os.path.join(UPLOAD_FOLDER, fn_2)
    if is_match:
        result = "两幅图片完全匹配。"
    else:
        result = "两幅图片不匹配。"
    elapsed = "耗时: {:.4f} 秒".format(elapsed)
    return render_template('result_match_image.html', result=result, elapsed=elapsed)


@app.route('/video-match')
def video_match():
    return render_template('match_video.html')


@app.route('/video-match-api')
def video_match_api():
    return render_template('match_video_api.html')


@app.route('/process_video_match', methods=['POST'])
def process_video_match():
    name, prob, index, time_in_video, elapsed, upload_file, image_fn = match_video()
    upload_file = os.path.join(UPLOAD_FOLDER, upload_file)
    result = "匹配度最高的广告：{}。".format(name)
    frame_index = "帧数: {}".format(index)
    time_in_video = "秒数: {:.2f} 秒".format(time_in_video)
    prob = '置信度: {:.4f}'.format(prob)
    elapsed = "耗时: {:.4f} 秒".format(elapsed)

    return render_template('result_match_video.html', result=result, frame_index=frame_index,
                           time_in_video=time_in_video, prob=prob, elapsed=elapsed,
                           upload_file=upload_file.split(".")[0] + "_adjust.jpg",
                           screenshot=image_fn)


@app.route('/process_video_match_api', methods=['POST'])
def process_video_match_api():
    name, prob, index, time_in_video, elapsed, upload_file, image_fn = match_video()
    upload_file_path = "http://47.101.196.204:8080/{}".format(UPLOAD_FOLDER)
    screen_shot_file = "http://47.101.196.204:8080/{}".format(image_fn)
    upload_file = os.path.join(upload_file_path, upload_file.split(".")[0] + "_adjust.jpg")
    response_result = jsonify(
        {'ad_name': name, 'frame_number': index, 'time_in_video': time_in_video, 'match_probablity': prob,
         'elapsed_time': elapsed, 'upload_file': upload_file, 'screen_shot_file': screen_shot_file})
    response_result.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response_result


@app.route('/tag_search')
def tag_search():
    return render_template('tag_search.html')


@app.route('/process_tag_search', methods=['POST'])
def process_tag_search():
    mac, gender, age, zcdj, yf, yc, intr = search_tag()
    return render_template('result_tag_search.html', mac=mac, gender=gender, age=age, zcdj=zcdj, yf=yf, yc=yc,
                           intr=intr)


@app.route('/asr', methods=['GET'])
def asr():
    return render_template('asr.html')


@app.route('/process_asr', methods=['POST'])
def process_asr():
    text = do_recognize()
    return render_template('result_asr.html', result=text)


@app.route('/tts', methods=['GET'])
def tts():
    return render_template('tts.html')


@app.route('/process_tts', methods=['POST'])
def process_tts():
    return render_template('result_tts.html', audiopath='')


@app.route('/sdk')
def sdk():
    return render_template('sdk.html')


@app.route('/solution')
def solution():
    return render_template('solution.html')


@app.route('/price')
def price():
    return render_template('price.html')


@app.route('/developer')
def developer():
    return render_template('developer.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
