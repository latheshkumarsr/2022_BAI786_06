
from flask import Flask

def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')
    from . import routes, api
    app.register_blueprint(routes.bp)
    app.register_blueprint(api.bp, url_prefix='/api')
    return app

if __name__ == '__main__':
    create_app().run(debug=True)
