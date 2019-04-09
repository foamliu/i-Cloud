from flask import Flask
from flask_bootstrap import Bootstrap


def create_app(config_name):
    app = Flask(config_name, static_url_path="", static_folder="static")
    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api/v1')
    Bootstrap(app)
    return app
