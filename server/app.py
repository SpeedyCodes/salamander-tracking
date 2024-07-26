from flask import Flask, request, Response
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Performs recognition and returns candidates with their confidence levels.
    The submitted image is saved to be confirmed later.
    """
    image = request.data
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite('image.jpg', image)
    return {
        "transient_id": 1,
        "candidates": [
            {
                "name": "John Doe",
                "id": 1, 
                "confidence": 0.9
            },
            {
                "name": "Jane Doe",
                "id": 2,
                "confidence": 0.8
            }
        ]
    }
    
@app.route('/confirm/<int:transient_id>', methods=['POST'])
def confirm():
    """
    Confirms the identity of the candidate with the given transient_id.
    """
    
    existing_id = request.json['id']
    return
    
@app.route('/register_new/<int:transient_id>', methods=['POST'])
def register_new():
    """
    Registers a new salamander with the given transient_id.
    """
    
    return

@app.route('/info/<int:id>', methods=['POST'])
def info(id):
    """
    Returns the information of the salamander with the given id.
    """    
    
    return Response(cv2.imencode('.jpg', cv2.imread('image.jpg'))[1].tobytes(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")