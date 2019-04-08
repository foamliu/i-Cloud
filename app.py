# flask_web/app.py
import os
import time

from flask import Flask
from flask import render_template, request
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from utils import compare, ensure_folder, FaceNotFoundError, resize

app = Flask(__name__, static_url_path="", static_folder="static")


@app.route('/')
def upload():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        start = time.time()
        ensure_folder('static')
        file1 = request.files['file1']
        filename_1 = secure_filename(file1.filename)
        full_path_1 = os.path.join('static', filename_1)
        file1.save(full_path_1)
        resize(full_path_1)
        file2 = request.files['file2']
        filename_2 = secure_filename(file2.filename)
        full_path_2 = os.path.join('static', filename_2)
        file2.save(full_path_2)
        resize(full_path_2)

        try:
            prob, is_same = compare(full_path_1, full_path_2)
            elapsed = time.time() - start
            if is_same:
                result = "验证结果：两张脸属于同一个人。"
            else:
                result = "验证结果：两张脸属于不同的人。"
            prob = "置信度为 {:.5f}".format(prob)
            elapsed = "耗时: {:.2f} 秒".format(elapsed)
        except FaceNotFoundError as err:
            result = '对不起，[{}] 图片中没有检测到人类的脸。'.format(err)
            prob = ""
            elapsed = ""

        return render_template('show.html', result=result, filename_1=filename_1, filename_2=filename_2, prob=prob,
                               elapsed=elapsed)


@app.route('/solution')
def solution():
    return render_template('solution.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    bootstrap = Bootstrap(app)
    app.run(host='0.0.0.0', port=8080, threaded=True)
