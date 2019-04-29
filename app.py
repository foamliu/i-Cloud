# flask_web/app.py

from flask import Flask
from flask import render_template
from flask_bootstrap import Bootstrap

from face_utils import face_verify
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
    return render_template('face-detect.html')


@app.route('/', methods=['GET'])
def home():
    return render_template('face-verify.html')


@app.route('/verify', methods=['GET'])
def verify():
    return render_template('face-verify.html')


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

    return render_template('verify_result.html', result=result, fn_1=fn_1, fn_2=fn_2, prob=prob, elapsed=elapsed)


@app.route('/search')
def search():
    return render_template('face-search.html')


@app.route('/emotion')
def emotion():
    return render_template('emotion.html')


@app.route('/match')
def match():
    return render_template('video_match.html')


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
