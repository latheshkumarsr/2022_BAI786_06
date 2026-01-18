
from flask import Blueprint, jsonify, request

bp = Blueprint('api', __name__)

@bp.route('/heatmap')
def heatmap():
    demo = {
        "type":"FeatureCollection",
        "features":[]
    }
    return jsonify(demo)

@bp.route('/top-cells')
def top_cells():
    return jsonify([{"cell_id":"A-101","risk":0.9,"count":10}])

@bp.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    return jsonify({"risk": 0.5, "input": data})
