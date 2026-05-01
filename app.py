import os
import uuid

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from src.pipeline import StyleTransferPipeline
from src.config import (
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_DIM,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_TV_WEIGHT,
    DEFAULT_LR,
    DEFAULT_SAVE_EVERY
)


app = Flask(__name__)

app.config["UPLOAD_FOLDER_CONTENT"] = "assets/content"
app.config["UPLOAD_FOLDER_STYLE"] = "assets/style"
app.config["OUTPUT_FOLDER"] = "outputs"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


for folder in [
    app.config["UPLOAD_FOLDER_CONTENT"],
    app.config["UPLOAD_FOLDER_STYLE"],
    app.config["OUTPUT_FOLDER"]
]:
    os.makedirs(folder, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """
    Checks if an uploaded file has an allowed image extension.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template(
        "index.html",
        defaults={
            "iterations": DEFAULT_ITERATIONS,
            "style_weight": DEFAULT_STYLE_WEIGHT,
            "content_weight": DEFAULT_CONTENT_WEIGHT,
            "tv_weight": DEFAULT_TV_WEIGHT,
            "lr": DEFAULT_LR,
            "max_dim": DEFAULT_MAX_DIM
        }
    )


@app.route("/generate", methods=["POST"])
def generate():
    try:
        content_file = request.files.get("content_img")
        style_files = request.files.getlist("style_imgs")

        if content_file is None or content_file.filename == "":
            return jsonify({"error": "Missing content image"}), 400

        if not style_files or style_files[0].filename == "":
            return jsonify({"error": "Missing style image"}), 400

        if not allowed_file(content_file.filename):
            return jsonify({"error": "Unsupported content image format"}), 400

        for style_file in style_files:
            if style_file.filename and not allowed_file(style_file.filename):
                return jsonify({"error": "Unsupported style image format"}), 400

        content_name = secure_filename(
            f"{uuid.uuid4()}_{content_file.filename}"
        )

        content_path = os.path.join(
            app.config["UPLOAD_FOLDER_CONTENT"],
            content_name
        )

        content_file.save(content_path)

        style_paths = []

        for style_file in style_files:
            if style_file.filename:
                style_name = secure_filename(
                    f"{uuid.uuid4()}_{style_file.filename}"
                )

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER_STYLE"],
                    style_name
                )

                style_file.save(style_path)
                style_paths.append(style_path)

        config = {
            "iterations": int(request.form.get("iterations", DEFAULT_ITERATIONS)),
            "style_weight": float(request.form.get("style_weight", DEFAULT_STYLE_WEIGHT)),
            "content_weight": float(request.form.get("content_weight", DEFAULT_CONTENT_WEIGHT)),
            "tv_weight": float(request.form.get("tv_weight", DEFAULT_TV_WEIGHT)),
            "lr": float(request.form.get("lr", DEFAULT_LR)),
            "start_from": request.form.get("start_from", "content"),
            "max_dim": int(request.form.get("max_dim", DEFAULT_MAX_DIM)),
            "save_every": DEFAULT_SAVE_EVERY
        }

        output_filename = f"output_{uuid.uuid4().hex}.png"

        output_path = os.path.join(
            app.config["OUTPUT_FOLDER"],
            output_filename
        )

        pipeline = StyleTransferPipeline(config)
        result = pipeline.run(content_path, style_paths, output_path)

        return jsonify({
            "status": "success",
            "output_image": output_filename,
            "time": round(result["elapsed_time"], 2),
            "loss_csv": result.get("loss_csv"),
            "loss_plot": result.get("loss_plot"),
            "intermediate_images": result.get("intermediate_images", [])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/result")
def result():
    img = request.args.get("img")
    time_taken = request.args.get("time")
    loss_csv = request.args.get("loss_csv")
    loss_plot = request.args.get("loss_plot")

    return render_template(
        "result.html",
        img=img,
        time=time_taken,
        loss_csv=loss_csv,
        loss_plot=loss_plot
    )


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


if __name__ == "__main__":
    # For final demo, debug=False and threaded=False are more stable
    # with TensorFlow and matplotlib on Windows.
    app.run(debug=False, threaded=False)