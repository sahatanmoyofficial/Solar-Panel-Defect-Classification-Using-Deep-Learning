import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="centered"
)

st.title("☀️ Solar Panel Defect Classifier")
st.write("Upload an image of a solar panel to detect defects using your model")

# Loading the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trained_effnet_finetune.h5")
    return model

with st.spinner("Loading model..."):
    model = load_model()
   
CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",             
    "Electrical-damage", 
    "Physical-damage",
    "Snow-Covered"
]

# File uploader
uploaded_file = st.file_uploader("Upload a solar panel image..", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
   
    # Preprocess for EfficientNet
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
   
    with st.spinner("Analyzing the panel..."):
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
       
    predicted_class = CLASSES[predicted_idx]
   
    st.markdown(f"""
                ### **Prediction: {predicted_class}**
                ***Confidence: {confidence:.1%}**
                """)
               
    if predicted_class == "Clean":
        st.success("The panel appears to be in good condition!!")
    else:
        st.warning("A defect or contamination has been detected!!")
   
   
    # Top 3 predictions
    st.write("### Top 3 Predictions")
    top_indices = np.argsort(predictions[0])[-3:][::-1]
   
    # Display top 3 with class names and confidence
    for i, idx in enumerate(top_indices):
        class_name = CLASSES[idx]
        prob = predictions[0][idx]
        # Color code: green for 1st, blue for 2nd, orange for 3rd
        if i == 0:
            st.markdown(f"**1st**: **{class_name}** - {prob:.1%}")
        elif i == 1:
            st.markdown(f"**2nd**: {class_name} - {prob:.1%}")
        else:
            st.markdown(f"**3rd**: {class_name} - {prob:.1%}")
            
            
    with st.expander("View all class probabilities"):
        for i, prob in enumerate(predictions[0]):
            st.write(f"{CLASSES[i]:<20} {prob:.1%}")