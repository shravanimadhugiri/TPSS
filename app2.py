import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# Load the model
model = tf.keras.models.load_model("resnet_model2.keras")  # Replace with your model path

# Feedback Storage File
feedback_file = "feedback.csv"

# Ensure the feedback file exists
if not os.path.exists(feedback_file):
    pd.DataFrame(columns=["Image Name", "Prediction", "Feedback"]).to_csv(feedback_file, index=False)

# Predict function
def predict_image_class(image):
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_labels = ["Positive", "Negative", "IR"]  # Modify as per your class names
    return class_labels[class_idx]

# Page Configuration
st.set_page_config(
    page_title="TPT App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header Section with Logos
cols = st.columns(3)


# Load your images
logo1 = Image.open("jssateb_logo.jpeg.jpg")
logo2 = Image.open("4.png.jpg")
logo3 = Image.open("jssaher_logo.png.jpg")

# Display logos above the title
#cols[0].image(logo1, use_container_width=True)
cols[1].image(logo2, use_container_width=True)
#cols[2].image(logo3, use_container_width=True)

#st.sidebar.image(logo1, caption="JSSATEB", use_container_width=True)
#st.sidebar.image(logo3, caption="JSSAHER", use_container_width=True)

# Sidebar for Extra Features
with st.sidebar:
    # Add logos above "Explore More"
    col1, col2 = st.columns(2)  # Create two columns for logos
    
    with col1:
        st.image(logo1, use_container_width=True)
    with col2:
        st.image(logo3, use_container_width=True)


# Title and Subtitle Section
st.markdown(
    """
    <style>
        .header-container {
            text-align: center;
            background: linear-gradient(to right, #00c0de, #3c9ba9);
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: white;
        }
        .header-subtitle {
            font-size: 1.2rem;
            color: #f0f0f0;
        }
    </style>
    <div class="header-container">
        <div class="header-title">TPT App</div>
        <div class="header-subtitle">Upload an image of the patch test of Allergic Contact Dermatitis to classify it as Positive, Negative, or IR</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Explore More")

home_button = st.sidebar.button("Home")
project_details_button = st.sidebar.button("Project Details")
#contact_us_button = st.sidebar.button("Contact Us")

if home_button:
    option = "Home"
elif project_details_button:
    option = "Project Details"
#elif contact_us_button:
    #option = "Contact Us"
else:
    option = "Home"  # Default option


# Project Details Page
if option == "Project Details":
    st.subheader("Project Details")
    st.markdown("""
        ### Objective
        This project aims to classify images of the patch test into predefined categories (Positive, Negative, and IR) using a deep learning model. 
        It provides an intuitive interface for users to upload images and receive predictions, along with an option to provide feedback for continuous improvement.
        
        ### Technologies Used:
        - TensorFlow for deep learning model
        - Streamlit for building the web app
        - PIL for image processing
        - Pandas for feedback management

        ### How it works:
        - The user uploads an image.
        - The app preprocesses the image and classifies it using the trained MobileNet model.
        - The prediction is displayed along with an option to submit feedback.
        - Feedback is stored in a structured CSV file for analysis and model enhancement.
    """)
    st.markdown("For more information, contact us or refer to the documentation.")

# Contact Us Page
#if option == "Contact Us":
    #st.subheader("Contact Us")
    #st.markdown("""
        ### Placeholder Contact Information:
        #- **Email**: pratyusha.ry2003@gmail.com
        #- **Phone**: +1-234-567-890
        #- **Address**: Your Company, 1234 Street, City, Country
    #""")
    #st.markdown("We look forward to hearing from you!")

# Home Page
if option == "Home":
    # Image Upload Section
    uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing the image..."):
            # Predict and display result
            prediction = predict_image_class(image)
            st.success(f"### Prediction: **{prediction}**")

        # Add interactive feedback
        st.markdown("#### Was the prediction accurate?")
        feedback = st.radio("", ["Yes", "No"], index=0, horizontal=True)

        # Save feedback to CSV
        if st.button("Submit Feedback"):
            with open(feedback_file, "a") as f:
                feedback_data = f"{uploaded_file.name},{prediction},{feedback}\n"
                f.write(feedback_data)
            st.success("Thank you for your feedback!")

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #aaa;
        }
    </style>
    <div class="footer">
        Developed by Lahari | Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
