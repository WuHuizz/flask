
from App.views.firstblue import blue
from App.views.secondblue import second
from App.views.thirdblue import third_blue
def init_view(app):
    app.register_blueprint(blue)
    app.register_blueprint(second)
    app.register_blueprint(third_blue)



