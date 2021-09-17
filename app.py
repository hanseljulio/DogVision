from enum import unique
from flask import Flask, render_template, redirect, url_for, request, session, flash
import tensorflow as tf
import tensorflow_hub as hub
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# app.config["IMAGE_UPLOADS"] = "static/uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPG", "PNG", "JFIF", "JPEG"]

# Define constant sizes
IMG_SIZE = 224
BATCH_SIZE = 32

unique_breeds = ['Affenpinscher', 'Afghan Hound', 'African hunting dog', 'Airedale',
       'American_staffordshire_terrier', 'Appenzeller',
       'Australian terrier', 'Basenji', 'Basset', 'Beagle',
       'Bedlington terrier', 'Bernese mountain dog',
       'Black and tan coonhound', 'Blenheim spaniel', 'Bloodhound',
       'Bluetick', 'Border collie', 'Border terrier', 'Borzoi',
       'Boston bull', 'Bouvier des flandres', 'Boxer',
       'Brabancon griffon', 'Briard', 'Brittany spaniel', 'Bull mastiff',
       'Cairn', 'Cardigan', 'Chesapeake bay retriever', 'Chihuahua',
       'Chow', 'Clumber', 'Cocker_spaniel', 'Collie',
       'Curly-coated retriever', 'Dandie dinmont', 'Dhole', 'Dingo',
       'Doberman', 'English foxhound', 'English setter',
       'English springer', 'Entlebucher', 'Eskimo dog',
       'Flat-coated retriever', 'French bulldog', 'German shepherd',
       'German short-haired pointer', 'Giant schnauzer',
       'Golden retriever', 'Gordon setter', 'Great dane',
       'Great pyrenees', 'Greater Swiss mountain dog', 'Groenendael',
       'Ibizan hound', 'Irish setter', 'Irish terrier',
       'Irish water spaniel', 'Irish wolfhound', 'Italian greyhound',
       'Japanese spaniel', 'Keeshond', 'Kelpie', 'Kerry blue terrier',
       'Komondor', 'Kuvasz', 'Labrador retriever', 'Lakeland terrier',
       'Leonberg', 'Lhasa', 'Malamute', 'Malinois', 'Maltese dog',
       'Mexican hairless', 'Miniature pinscher', 'Miniature poodle',
       'Miniature schnauzer', 'Newfoundland', 'Norfolk terrier',
       'Norwegian elkhound', 'Norwich terrier', 'Old english sheepdog',
       'Otterhound', 'Papillon', 'Pekinese', 'Pembroke', 'Pomeranian',
       'Pug', 'Redbone', 'Rhodesian ridgeback', 'Rottweiler',
       'Saint_bernard', 'Saluki', 'Samoyed', 'Schipperke',
       'Scotch terrier', 'Scottish deerhound', 'Sealyham terrier',
       'Shetland sheepdog', 'Shih-tzu', 'Siberian husky', 'Silky terrier',
       'Soft-coated wheaten terrier', 'Staffordshire bullterrier',
       'Standard poodle', 'Standard schnauzer', 'Sussex spaniel',
       'Tibetan mastiff', 'Tibetan terrier', 'Toy poodle', 'Toy terrier',
       'Vizsla', 'Walker hound', 'Weimaraner', 'Welsh springer spaniel',
       'West highland white terrier', 'whippet',
       'Wire-haired fox terrier', 'Yorkshire terrier']

# Deep learning functions

# Create function to load a trained model
# Loads a saved model from a specified path.
def load_model(model_path):
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
  return model


# Create a simple function to return a tuple (image, label)
# Takes an image file path name and the associated label, processes
# the image and return a tuple of (image, label)
def get_image_label(image_path, label):
  image = process_image(image_path)
  return image, label


# Check file extension
def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


# Turns an array of prediction probabilities into a label
def get_pred_label(prediction_probabilities):
  return unique_breeds[np.argmax(prediction_probabilities)]

# Process image function to be compatible for deep learning
def process_image(image_path, img_size=IMG_SIZE):
  # Read image file
  image = tf.io.read_file(image_path)

  # Turn jpg image into numerical Tensor with 3 color channels (RGB)
  image = tf.image.decode_jpeg(image, channels=3)

  # Convert the color channel values from 0-255 to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)

  # Resize the image to our desired value (224, 224)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

  return image


# Create a function to turn data into batches
# Creates batches of data out of image (X) and label (y) pairs.
# It shuffles the data if it's training data but doesn't shuffle if it's validation data.
# Also accepts test data as input (no labels)
def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  # If the data is a test data set, we probably don't have labels
  if test_data:
    print("Creating test data batches...")
    # Only filepaths (no labels)
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  
  # If the data is a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    # tf.constant(X) = filepaths, tf.constant(y) = labels
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))

    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(X))
    
    # Create (image, label) tuples (this also turns the image path into a preprocessed image)
    data = data.map(get_image_label)

    # Turn the training data into batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch


# Website functions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        if request.files:
            image = request.files['image']
            if image.filename == "":
                return '''
                <script> window.alert("Image must have a filename!"); </script>
                <script> window.location=document.referrer; </script>
                '''
            
            if not allowed_image(image.filename):
                return '''
                <script> window.alert("Image extension is not allowed!"); </script>
                <script> window.location=document.referrer; </script>
                '''
            else:
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.root_path,'static/uploads/', filename))
                return redirect(url_for('result', filename=filename))
            
    return render_template('index.html')

@app.route('/list', methods=['GET', 'POST'])
def doglist():
    return render_template('doglist.html', unique_breeds=unique_breeds, length=len(unique_breeds))

@app.route('/result', methods=['GET', 'POST'])
def result():
    filename = "static/uploads/" + request.args.get("filename")
    model = load_model("21-21211630876899-full-image-set-mobilenetv2-Adam.h5")

    filename_paths = [filename]

    custom_data = create_data_batches(filename_paths, test_data=True)

    # Make predictions on custom data
    custom_preds = model.predict(custom_data)

    # Get custom image prediction labels
    custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
    result = custom_pred_labels[0]

    return render_template('result.html', filename=filename, result=result)

if __name__ == '__main__':
    app.run(debug=False)