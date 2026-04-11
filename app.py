import os
import uuid
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from src.pipeline import StyleTransferPipeline
from src.config import DEFAULT_ITERATIONS, DEFAULT_MAX_DIM

app = Flask(__name__)
app.config['UPLOAD_FOLDER_CONTENT'] = 'assets/content'
app.config['UPLOAD_FOLDER_STYLE'] = 'assets/style'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # הגבלת משקל תמונה ל-16MB


# וידוא תיקיות
for folder in [app.config['UPLOAD_FOLDER_CONTENT'], app.config['UPLOAD_FOLDER_STYLE'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        content_file = request.files['content_img']
        style_files = request.files.getlist('style_imgs') # תמיכה ב-Multi-style
        
        if not content_file or not style_files[0]:
            return jsonify({"error": "Missing images"}), 400

        # שמירת תמונות בצורה בטוחה
        content_name = secure_filename(f"{uuid.uuid4()}_{content_file.filename}")
        content_path = os.path.join(app.config['UPLOAD_FOLDER_CONTENT'], content_name)
        content_file.save(content_path)
        
        style_paths = []
        for sf in style_files:
            if sf.filename:
                s_name = secure_filename(f"{uuid.uuid4()}_{sf.filename}")
                s_path = os.path.join(app.config['UPLOAD_FOLDER_STYLE'], s_name)
                sf.save(s_path)
                style_paths.append(s_path)

        # קריאת פרמטרים מהמשתמש
        config = {
            'iterations': int(request.form.get('iterations', DEFAULT_ITERATIONS)),
            'style_weight': float(request.form.get('style_weight', 1e-2)),
            'content_weight': float(request.form.get('content_weight', 1.0)),
            'tv_weight': float(request.form.get('tv_weight', 1e-4)),
            'start_from': request.form.get('start_from', 'content'),
            'max_dim': DEFAULT_MAX_DIM
        }

        output_filename = f"output_{uuid.uuid4()}.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # הרצת האלגוריתם
        pipeline = StyleTransferPipeline(config)
        result = pipeline.run(content_path, style_paths, output_path)

        return jsonify({
            "status": "success",
            "output_image": output_filename,
            "time": round(result['elapsed_time'], 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result')
def result():
    img = request.args.get('img')
    time_taken = request.args.get('time')
    return render_template('result.html', img=img, time=time_taken)

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)