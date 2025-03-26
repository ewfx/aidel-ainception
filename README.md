# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
AI based Entity Intelligence and Risk Scoring. Develop a sophisticated generative ai/ml powered system that automates the research and evidence-gathering process for identifying verifying and risk scoring entites from complex multi source transaction data.
## 🎥 Demo
📹 [Video Demo](#) was attached in the Artifacts

🖼️ Screenshots:
![image](https://github.com/user-attachments/assets/44ad5957-11fe-4ac9-bc0e-9b99b3edf00f)


## ⚙️ What It Does
Our model takes input transactions as a csv or text file and our entity extraction model then runs and gives us the entities involved in the transaction. We're using a Entity Classifier to classify the type of entities involved in the transaction. Risk of transaction will be calculated based on Transaction Amount, Payer and Receiver Geo Location, along with Payer and Receiver past history details form several open source datasets. Anomaly Entities are being identified and given high risk. Along with risk score, we're assigning a risk category. Confidence score is calculated based on what sources we're asigning particular risk and the same will be mentioned in Supporting Evidence and Justification Text.

## 🛠️ How We Built It
Streamlit - UI
BERT - Entity Extraction
Sentiment Analysis
Pandas - DataFrames
Spacy 
Uvicorn
HuggingFace Transformers
FastAPI

## 🚧 Challenges We Faced
OpenCorporates was not accessible during the hackathon 

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/your-repo.git
   ```
2. Install dependencies  
   ```sh
   npm install  # or pip install -r requirements.txt (for Python)
   ```
3. Run the project  
   ```sh
   npm start  # or python app.py
   ```

## 🏗️ Tech Stack
- 🔹 Frontend: Streamlit
- 🔹 Backend: Python, Fast API
- 🔹 Models used:
     🔹https://huggingface.co/deepset/bert-base-cased-squad2
     🔹https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2


## 👥 Team
- **Thilaksurya B**
- **Susmitha Priya Maddula**
- **Rwitick Ghosh**
- **Aakash Ravi**
- **Ankita Singh**
