# Sentiment Analysis using SVM & TF-IDF  

## 📌 Project Overview  
This project is a **Sentiment Analysis System** built using **Support Vector Machine (SVM)** and **TF-IDF Vectorization**.  
It classifies text data into **positive or negative sentiments** based on a given dataset.  

The project is implemented in **Jupyter Notebooks** and is designed to run seamlessly in **Google Colab** with data and models stored in Google Drive.  

---

## ⚙️ Tech Stack & Requirements  
- **Python 3.x**  
- **Google Colab** (preferred)  
- **Libraries Used:**  
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib (optional for visualization)  

No local installation is required since everything can be run inside **Colab**.  

---

## 📂 Project Structure  

```
sentemental_analysis/
│── App.ipynb                  # Main notebook to run predictions
│── Sentemental_analysis.ipynb # Notebook for training & evaluation
│── Dataset_changed.csv        # Dataset in CSV format
│── Dataset_changed.xlsx       # Dataset in Excel format
│── svm_model.pkl              # Trained SVM model
│── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
```

---

## 🛠️ Setup Instructions  

Follow these steps to run the project in **Google Colab**:  

1. **Download or Clone the Repository**  
   - Extract the `sentemental_analysis` folder from the provided project zip.  
   - Upload the folder into your **Google Drive**.  

2. **Open Google Colab**  
   - Go to [Google Colab](https://colab.research.google.com/).  
   - Open either `App.ipynb` (for predictions) or `Sentemental_analysis.ipynb` (for training).  

3. **Mount Google Drive in Colab**  
   At the top of the notebook, run the following:  
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Navigate to Project Folder**  
   Adjust the path according to where you placed the folder in your Drive:  
   ```python
   import os
   project_path = "/content/drive/MyDrive/sentemental_analysis"
   os.chdir(project_path)
   ```

5. **Install Required Libraries (if missing)**  
   ```python
   !pip install scikit-learn pandas numpy
   ```

---

## 🚀 Usage Guide  

### 1. **Training the Model** (optional)  
- Open **`Sentemental_analysis.ipynb`**  
- Run all cells → This will:  
  - Load the dataset  
  - Preprocess text data  
  - Train the SVM model  
  - Save the trained model as `svm_model.pkl` and the vectorizer as `tfidf_vectorizer.pkl`  

### 2. **Running Predictions**  
- Open **`App.ipynb`**  
- This notebook will:  
  - Load the pre-trained model (`svm_model.pkl`)  
  - Load the saved TF-IDF vectorizer (`tfidf_vectorizer.pkl`)  
  - Accept text input  
  - Predict whether the sentiment is **Positive** or **Negative**  

Example:  
```python
text = ["I really enjoyed this product!"]
prediction = model.predict(vectorizer.transform(text))
print(prediction)  # Output: Positive
```

---

## 🧪 Testing  
- Try different sentences and see how the model classifies them.  
- Modify the dataset (`Dataset_changed.csv`) and retrain if needed.  

---

## 📑 Additional Notes  
- If you face path issues in Colab, double-check the folder path in Drive.  
- You can extend this project by:  
  - Adding more preprocessing (stopword removal, stemming, lemmatization)  
  - Using a larger dataset  
  - Trying other ML models (Logistic Regression, Naive Bayes, etc.)  
- This project is meant for **educational purposes** and can be a foundation for more advanced NLP tasks.  
