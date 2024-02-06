import streamlit as st
import pickle
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model

token=0
def add_prediction(data):
    # model=pickle.load(open("model/model.pkl", "rb"))
    model = load_model("model/model2.h5")
    prediction = model.predict(data)
    predicted_class_index = np.argmax(prediction)
    
       
    with st.container():
        st.markdown(f"<h1 class=result>{predicted_class_index}</h1>",unsafe_allow_html=True )
    
    
    
def canvas_component():
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=12,
        stroke_color='#999999',
        background_color='#000000',
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode='freedraw',
        point_display_radius=0,
        key="full_app",
    )
    
    if canvas_result.image_data is not None:
        
        img_array = np.array(canvas_result.image_data)
        resized_img = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_CUBIC)[:,:,0]
        st.subheader("")
        st.subheader("Recognized Digit") 
        if(resized_img.any()!=0):
          normalized_img = resized_img
          final_img=(np.array([normalized_img]))
          add_prediction(final_img)
    
    
def main():
    st.set_page_config(
        page_title="Digit Recognition",
        page_icon="::",
        layout="centered",
    )
    
    with open("assets/styles.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    with st.container():
        st.title("Digit Recognition")
        st.markdown("Using Convolutional Neural Network")
        with st.container():
            canvas_component()
        
        
if __name__ == '__main__':
    main()