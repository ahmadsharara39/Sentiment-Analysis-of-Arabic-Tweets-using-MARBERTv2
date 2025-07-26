from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from routes.model_routes import model_bp
app.register_blueprint(model_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)

