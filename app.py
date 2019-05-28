# flask_web/app.py

from flask import Flask
from flask import render_template
from flask_bootstrap import Bootstrap

from utils import FaceNotFoundError
from utils_face import face_detect, face_verify, face_search
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
    num_faces, elapsed, fn = face_detect()
    if num_faces > 0:
        result = "图片中已检测到人脸。"
    else:
        result = "图片中没有检测到人类的脸。"
    num_faces = "人脸数量: {}".format(num_faces)
    elapsed = "耗时: {:.4f} 秒".format(elapsed)
    return render_template('result_detect.html', result=result, num_faces=num_faces, fn=fn, elapsed=elapsed)


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


@app.route('/emotion')
def emotion():
    return render_template('emotion.html')


@app.route('/image-match')
def image_match():
    return render_template('match_image.html')


@app.route('/process_image_match', methods=['POST'])
def process_image_match():
    is_match, elapsed, fn_1, fn_2 = match_image()
    if is_match:
        result = "两幅图片完全匹配。"
    else:
        result = "两幅图片不匹配。"
    elapsed = "耗时: {:.4f} 秒".format(elapsed)
    return render_template('result_match_image.html', result=result, elapsed=elapsed)


@app.route('/video-match')
def video_match():
    return render_template('match_video.html')


@app.route('/process_video_match', methods=['POST'])
def process_video_match():
    name, prob, index, time_in_video, elapsed, upload_file, image_fn = match_video()
    result = "匹配度最高的广告：{}。".format(name)
    frame_index = "帧数: {}".format(index)
    time_in_video = "秒数: {:.2f} 秒".format(time_in_video)
    prob = '置信度: {:.4f}'.format(prob)
    elapsed = "耗时: {:.4f} 秒".format(elapsed)

    return render_template('result_match_video.html', result=result, frame_index=frame_index,
                           time_in_video=time_in_video, prob=prob, elapsed=elapsed, upload_file=upload_file,
                           screenshot=image_fn)


@app.route('/tag_search')
def tag_search():
    return render_template('tag_search.html')


@app.route('/process_tag_search', methods=['POST'])
def process_tag_search():
    mac, gender, age, zcdj, yf, yc, intr = search_tag()
    return render_template('result_tag_search.html', mac=mac, gender=gender, age=age, zcdj=zcdj, yf=yf, yc=yc,
                           intr=intr)


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
