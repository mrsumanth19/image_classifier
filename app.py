# app.py
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from predict import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        
        label, confidence = predict_image(filepath)
        return render_template('index.html', prediction=label, confidence=confidence, image=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
