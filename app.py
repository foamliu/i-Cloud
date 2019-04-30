# flask_web/app.py

from flask import Flask
from flask import render_template
from flask_bootstrap import Bootstrap

from utils_face import face_detect, face_verify
from utils_match import video_match
from utils import FaceNotFoundError

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


@app.route('/emotion')
def emotion():
    return render_template('emotion.html')


@app.route('/match')
def match():
    return render_template('video_match.html')


@app.route('/process_match', methods=['POST'])
def process_match():
    is_match, index, time_in_video, elapsed, fn = video_match()
    if is_match:
        result = "验证结果：图片在视频中已定位。"
    else:
        result = "验证结果：图片在视频中无法找到。"
    frame_index = "第几帧: {}".format(index)
    time_in_video = "第几秒: {:.2f} 秒".format(time_in_video)
    elapsed = "耗时: {:.4f} 秒".format(elapsed)

    return render_template('result_match.html', result=result, frame_index=frame_index, time_in_video=time_in_video,
                           elapsed=elapsed, fn=fn)


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
