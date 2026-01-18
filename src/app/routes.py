
from flask import Blueprint, render_template
bp = Blueprint('routes', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict')
def predict_page():
    return render_template('predict.html')

@bp.route('/admin')
def admin_page():
    return render_template('admin.html')
