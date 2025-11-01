import streamlit as st
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from streamlit_option_menu import option_menu
import joblib
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import plotly.express as px

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title='News Anomaly Dashboard', layout='wide')

# ---------------------------
# LOAD DATA AND MODELS
# ---------------------------
df = pd.read_csv(r'C:\Users\DEVA NANTHAN\Documents\ano\out.csv', encoding='latin1')

@st.cache_resource
def load_models():
    scaler = joblib.load("scaler1.pkl")
    clf = joblib.load("XGBClassifier1.pkl")              # anomaly model
    loc_encoder = joblib.load("LabelEncoder1.pkl")
    loc_model = joblib.load("XGBClassifier1.pkl")    # <— new model for location prediction
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return scaler, clf, loc_encoder, loc_model, embed_model

scaler, clf, loc_encoder, loc_model, embed_model = load_models()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ---------------------------
# SIDEBAR MENU
# ---------------------------
with st.sidebar:
    selected = option_menu(
        "Main Menu", ["Data", "New"],
        icons=['table', 'plus-circle'], menu_icon="cast", default_index=0
    )


    
    
    

    

if selected=="Data":
    inp=st.radio('choose',('metrics','Anamoly'),horizontal=True)
    if inp=='metrics':
        st.title('📰 News Article Anomaly Dashboard')
        
        sel_topic = st.sidebar.multiselect('Select Topic(s)', df['NewsType_final'].unique())
        if sel_topic:
            df = df[df['NewsType_final'].isin(sel_topic)]
        col1,col2=st.columns(2)
        with col1:
            
            st.metric('Total Articles', len(df))
        with col2:
            st.metric('Flagged Anomalies', df['is_anomaly'].sum())

        fig1 = px.histogram(df, x='sentiment', color='is_anomaly', title='Sentiment Distribution')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter(df, x='sentiment_score', y='anomaly_score',
                    color=df['is_anomaly'].map({True:'Anomaly',False:'Normal'}),
                    hover_data=['Heading','main_location'])
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader('Flagged Articles')
        st.dataframe(df[df['is_anomaly']==True][['Date','Heading','main_location','NewsType_final','sentiment','anomaly_score']])
        lf=df.describe()
        st.write(lf)
    
    if inp=='Anamoly':
        col1,col2,col3=st.columns(3)
        with col1:
            loc=st.multiselect('select location',df['main_location'].unique())
        with col2:
            top=st.multiselect('select topic',df['NewsType_final'].unique())
        with col3:
            det=st.multiselect('select anamoly',df['is_anomaly'].unique())

        with st.container():
            
            st.markdown('### Filtered Articles')

        # Apply filters one by one
            filtered_df = df.copy()

            if loc:
                filtered_df = filtered_df[filtered_df['main_location'].isin(loc)]
            if top:
                filtered_df = filtered_df[filtered_df['NewsType_final'].isin(top)]
            if det:
                filtered_df = filtered_df[filtered_df['is_anomaly'].isin(det)]

            # Display filtered articles
            st.write(filtered_df[['main_location', 'NewsType_final', 'is_anomaly', 'Article']])

        
                























# ---------------------------
# NEW ARTICLE PREDICTION PAGE
# ---------------------------
if selected == 'New':
    st.title("📰 New Article Anomaly Prediction (Auto Location)")

    article = st.text_area("✍️ Enter the news article text:", height=200)

    def preprocess_text(text):
        text = text.lower()
        tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
        return " ".join(tokens)

    if st.button("🔍 Predict"):
        if article.strip() == "":
            st.warning("Please enter some text before predicting.")
        else:
            clean_text = preprocess_text(article)
            text_embedding = embed_model.encode([clean_text])

            # 🔹 Step 1 — Predict Location
            loc_pred = loc_model.predict(text_embedding)[0]
            location_name = loc_encoder.inverse_transform([loc_pred])[0]

            # 🔹 Step 2 — Combine embedding + location for anomaly prediction
            combined_features = np.concatenate([text_embedding[0], [loc_pred]]).reshape(1, -1)

            # 🔹 Step 3 — Scale and predict anomaly
            if combined_features.shape[1] != scaler.n_features_in_:
                st.error(f"Feature mismatch! Scaler expects {scaler.n_features_in_}, but got {combined_features.shape[1]}")
            else:
                scaled_vector = scaler.transform(combined_features)
                prediction = clf.predict(scaled_vector)[0]
                proba = clf.predict_proba(scaled_vector)[0][1]

                st.markdown("---")
                st.subheader("📊 Prediction Result:")
                st.info(f"🌍 Predicted Location: **{location_name}**")

                if prediction == 1:
                    st.error(f"🚨 The article from **{location_name}** is **Anomalous**. Confidence: `{proba:.2f}`")
                else:
                    st.success(f"✅ The article from **{location_name}** is **Normal News**. Confidence: `{proba:.2f}`")

print("done")
