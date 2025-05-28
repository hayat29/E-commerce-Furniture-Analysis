# 📦 E-Commerce Furniture Price Prediction Project  

## 🏡 **Overview**  
This project is designed to **analyze and predict furniture prices** based on various features such as product descriptions, original price, number of sales, and tags. It leverages **machine learning**, **natural language processing (NLP)**, and **data visualization** to uncover pricing trends and predict optimal prices for new furniture listings.  

## 🚀 **Objectives**
- Clean and preprocess furniture e-commerce dataset  
- Extract **text-based features** using **TF-IDF Vectorization**  
- Encode categorical data  
- Train predictive models (**Linear Regression, Random Forest Regressor**)  
- Evaluate models using **MSE and R² scores**  

---

## 🔧 **Tech Stack & Dependencies**  
The project is built using **Python** with the following libraries:  

### **Core Libraries**
- `pandas` → Data manipulation  
- `numpy` → Numerical computations  
- `matplotlib`, `seaborn` → Data visualization  

### **Machine Learning & NLP**
- `sklearn.feature_extraction.text.TfidfVectorizer` → Extract features from product descriptions  
- `sklearn.preprocessing.LabelEncoder` → Encode categorical data  
- `sklearn.model_selection.train_test_split` → Split dataset  
- `sklearn.linear_model.LinearRegression` → Train linear regression model  
- `sklearn.ensemble.RandomForestRegressor` → Train random forest model  
- `sklearn.metrics.mean_squared_error, r2_score` → Evaluate performance  

---

## 📊 **Dataset Description**
The dataset consists of **e-commerce furniture listings** with the following columns:  

| Column | Description |
|--------|------------|
| `productTitle` | Name of the furniture item |
| `originalPrice` | Listed original price (if available) |
| `price` | Current selling price |
| `sold` | Number of items sold |
| `tagText` | Additional product tags (e.g., “Free Shipping”) |

🔹 **Handling Missing Values** → Missing prices are filled with the **median price**  
🔹 **Feature Engineering** → Text-based features extracted using **TF-IDF**  

---

## 🏆 **How to Run**
### **1️⃣ Install Dependencies**
Run this command in your terminal:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **2️⃣ Load Dataset**
Place your dataset (`e_commerce_furniture.csv`) inside the `data/` folder.

### **3️⃣ Train the Model**
Run the following Python script:  
```python
python src/train.py
```

### **4️⃣ Evaluate the Model**
Check model performance using:  
```python
python src/evaluate.py
```

---

## 📈 **Results & Insights**
- **Text Features Matter!** → Product descriptions impact pricing  
- **Random Forest Model Performs Best** → Lower MSE compared to Linear Regression  
- **More Sales = Lower Prices?** → Discounts increase sales but affect profitability  

---

## 👏 **Contributors**
- **Faisal** *(Lead Developer & Data Analyst)*  
---

### 💡 **Future Improvements**
🔹 **Integrate Deep Learning** → Use LSTMs for better NLP feature extraction  
🔹 **Recommendation System** → Suggest ideal furniture pricing strategies  
🔹 **Real-Time Predictions** → Deploy as an API for live price estimates  

