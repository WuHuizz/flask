from flask_script import Manager
from App import creat_app
app = creat_app()
manager = Manager(app=app)

if __name__ == '__main__':
    manager.run()

