# Hyperlocal-News-Anomaly-Detection-and-Source-Attribution
An advanced NLP-based system to detect fake or misleading hyperlocal news by analyzing linguistic patterns, sentiment, and location consistency.

Project Overview
An advanced NLP-based system to detect fake or misleading hyperlocal news by analyzing linguistic patterns, sentiment, and location consistency.  
It uses BERT/RoBERTa embeddings, topic modeling, NER-based location extraction, and anomaly detection (Isolation Forest, One-Class SVM, Variational Autoencoder).  
The system visualizes anomalies on an interactive Streamlit dashboard, deployable on AWS.

Objectives
- Identify linguistic anomalies in local news content.  
- Detect source misattribution using content-based location prediction.  
- Monitor temporal shifts in sentiment and topics across regions.  
- Provide interpretable results with explainable anomaly scores.

Key Features
-  Named Entity Recognition (NER) for location extraction.  
-  Sentiment analysis (VADER / RoBERTa).  
-  Topic modeling (BERTopic).  
-  Feature fusion with transformer embeddings + temporal signals.  
-  Multi-model anomaly detection (Isolation Forest).  
-  Interactive visualization (Streamlit).  
-  Cloud-ready deployment on AWS.

Tech Stack
Languages & Frameworks: Python, scikit-learn, PyTorch, TensorFlow  
Libraries: Hugging Face Transformers, SpaCy, BERTopic, VADER, Prophet  
Visualization: Streamlit, Plotly, Dash  

Dataset Description
| Column | Description |
|---------|-------------|
| `Article` | Full news article text |
| `Heading` | Title or headline |
| `Date` | Publication date |
| `NewsType` | Type of news (business, politics, etc.) |
| `Source` | Publisher or source info |
