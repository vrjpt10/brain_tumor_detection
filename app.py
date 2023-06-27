import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('brain_tumor_epoch_100.h5')

# Define the tumor types
tumor_types = ['glioma_tumor', 'meningioma_tumor',
               'no_tumor', 'pituitary_tumor']

# Create the web application using Streamlit
st.title("Tumor Type Classification")
st.write("Upload an image and let the model predict the tumor type.")

# Display file upload widget
uploaded_file = st.file_uploader(
    "Choose an image...", type=['jpg', 'jpeg', 'png'])

# Make predictions when an image is uploaded
if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)

    # Preprocess the image
    # Resize the image to match the input size of the model
    image = image.resize((150, 150))
    image = tf.keras.preprocessing.image.img_to_array(image)
    # image) / 150.0  # Normalize pixel values to [0, 1]
    # image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    indices = prediction. argmax()
    predicted_class = tumor_types[int(indices)]

    # Display the uploaded image and predicted tumor type
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Tumor Type: {predicted_class}")
