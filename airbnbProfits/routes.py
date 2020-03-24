from flask import Blueprint, request
from .mlmodels import train, predict, loadModel

bp = Blueprint('routes', __name__, url_prefix='/api')


@bp.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'


@bp.route('/predictPrice', methods=['GET'])
def predict_price():
    return predict([[45, 45], [10, 50]])


@bp.route('/train', methods=['GET'])
def trainModel():
    result = train()
    return result


@bp.route('/loadModel', methods=['GET'])
def load():
    return loadModel()
