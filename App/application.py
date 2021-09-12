import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt


pipe = joblib.load(open("models/emotion_classifier_pipe_20210911.pkl", "rb"))

def predict_emotion(text):
    return pipe.predict([text])[0]


def predict_proba(text):
    return pipe.predict_proba([text])

emoji_dict = {"anger":"😡",
              "disgust":"🤮",
              "fear":"😱",
              "happy":"😊",
              "joy":"😍",
              "neutral":"😐",
              "sadness":"😢",
              "shame":"😓",
              "surprise":"😲"}


def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home - Emotion in Text")
        with st.form(key = "emotion_clf_form"):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label = "Submit")
        if submit_text:
            col1, col2 = st.columns(2)
            
            prediction = predict_emotion(raw_text)
            probability = predict_proba(raw_text)
            
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji = emoji_dict[prediction]
                st.write(f"{prediction}:{emoji}")
                st.write(f"Confidence: {np.max(probability)}")
            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe.classes_)
                #st.write(proba_df)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]
                
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x = "emotions", y = "probability", color = "emotions")
                st.altair_chart(fig, use_container_width = True)


    elif choice == "Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")


if __name__ == "__main__":
    main()

