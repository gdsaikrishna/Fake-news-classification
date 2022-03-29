import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageChops, ImageEnhance
import os
import numpy as np

def convert_to_ela_image(path, quality):
    temp_filename = r'static\images\temp_file_name.jpg'
    #temp_filename = os.path.join(app.root_path, app.config['STATIC_FOLDER'], 'images', 'temp_file_name.jpg')
    
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
    #ela_temp_filename = os.path.join(app.root_path, app.config['STATIC_FOLDER'], 'images', 'elatemp_file_name.jpg')
    ela_temp_filename = r'static\images\temp_file_name.jpg'
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


json_file = open('image_splice_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model

loaded_model.load_weights("image_splice_cnn_model.h5")
print("Loaded model from disk")


#fake_image_path = r'C:\Users\d.krishna.gundimeda\OneDrive - Accenture\Desktop\Fake Image Detector\label_in_wild\images\2251.jpg'
#print(image_prediction(fake_image_path))
#Image.open(image_path)
#true_image_path = r'download.jpg'
#print(image_prediction(true_image_path))