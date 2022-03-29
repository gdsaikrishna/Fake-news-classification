from flask import Flask, render_template,request
from forms import RegistrationForm
import os
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from text_explainer import text_predict_explain, predict

app = Flask(__name__)

app.debug = True

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['STATIC_FOLDER'] = 'static'

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route('/text', methods=['GET', 'POST'])
def text():
    explain = False
    output = ""
    if request.method == "POST":
        x = request.form.get("text", None)
        print(x)
        ans, percent = predict(x)
        if ans[0][0] == 0:
            percent = 100 - percent
            output = str(percent)+" % "+" Fake News \n"
        else:
            output = str(percent)+" % "+" Real News \n"
        print(output)
        text_predict_explain(x)
        explain=True
        
    filenames = ['accenture.jpg', 'accenture.jpg','accenture.jpg','accenture.jpg','accenture.jpg','accenture.jpg']    
    return render_template("my-form.html", filenames=filenames, explain=explain,link = r'C:\Users\d.krishna.gundimeda\Fake News Slyth\templates\text_explain.html',output= output)

@app.route('/image', methods=['GET', 'POST'])
def image():
    output= ""
    if request.method == "POST":
        print(request.files['img'])
        if request.files:
            image = request.files["img"]
            image.save(os.path.join('static/', image.filename))
            output = image_prediction(os.path.join('static/', image.filename))

    if request.method == "GET":
        output= ""

    return render_template("imagesearch.html", output = output)

@app.route('/dummy', methods=['GET', 'POST'])
def dummy():
    form = RegistrationForm()
    if request.method == "POST":
        x = request.form.get("text", None) 
        #filename = os.path.expanduser('.') + '/static/accenture.jpg'
        print(x)
    #url ="./static/accenture.jpg"

    return render_template("dumy2.html")

@app.route('/playground', methods=['GET', 'POST'])
def powerbi():

    return render_template("playground.html")


json_file = open('image_splice_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("image_splice_cnn_model.h5")
def convert_to_ela_image(path, quality):
    #temp_filename = r'static\images\temp_file_name.jpg'
    temp_filename = os.path.join(app.root_path, app.config['STATIC_FOLDER'], 'images', 'temp_file_name.jpg')
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_temp_filename = os.path.join(app.root_path, app.config['STATIC_FOLDER'], 'images', 'elatemp_file_name.jpg')
    ela_image.save(ela_temp_filename, 'JPEG')
    return ela_image

def prepare_image(image_path):
    image_size = (128, 128)
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

def image_prediction(path):
    image = prepare_image(path)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = loaded_model.predict(image)
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    output = ""
    if y_pred_class == 1:
        #print("Not Forged: Real Image")
        #print(f'Confidence: {np.amax(y_pred) * 100:0.2f}')
        confidence = str(round(np.amax(y_pred) * 100, 2))
        output = confidence + "% Real Image"
    else:
        #print("Forged: Fake Image")
        #print(f'Confidence: {np.amax(y_pred) * 100:0.2f}')
        confidence = str(round(np.amax(y_pred) * 100, 2))
        output = confidence + "% "+"Fake Image"
        print(output)
    return output


if __name__ == "__main__":
    app.run()
