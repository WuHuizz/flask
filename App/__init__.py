from flask import Flask
from App.views import  init_view
def creat_app():
    app = Flask(__name__)
    init_view(app)
    return app