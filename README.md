# 🧠 Chess Game Outcome Prediction

## 📍 Project Overview
- Dataset: 20,000+ chess games from **Lichess.org**
- Goal: Predict the game **outcome** (White win, Black win, or Draw)
- Features:
  - `rated`, `turns`, `victory_status`, `winner`, `increment_code`, `white_rating`, `black_rating`, `opening_name`, etc.

> The dataset is highly suitable for **Classification**, and even promising for **Reinforcement** or **Deep Learning** applications.

---

## 🧹 Data Cleaning & Preprocessing

### 🔧 Steps Taken from the Notebook:

```python
# Dropping non-informative columns
df.drop(['id', 'white_id', 'black_id'], axis=1, inplace=True)

# Converting 'rated' column to numeric
df['rated'] = df['rated'].astype(int)

# One-Hot Encoding categorical columns
df = pd.get_dummies(df, columns=['victory_status'], drop_first=False)

# Removing outliers in number of moves
Q1 = df['moves_num'].quantile(q=0.25)
Q3 = df['moves_num'].quantile(q=0.75)
IQR = Q3 - Q1
df.drop(df[df['moves_num'] >= Q3 + 1.5*IQR].index, inplace=True)

# Removing games with very few moves (low-quality/noise)
df.drop(df[df['moves_num'] < 8].index, inplace=True)

# Dropping time-related columns that are not useful
df.drop(['created_at', 'last_move_at', 'period'], axis=1, inplace=True)

# Attempting to clean the 'opening_name' field (was eventually dropped)
df['opening_name'] = df['opening_name'].apply(lambda x: x.split(':')[0].split('|')[0].split('#')[0])
df.drop(['opening_name'], axis=1, inplace=True)
```

---

## 🔍 Classification Models & Evaluation

### ✅ Models Used:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gaussian Naive Bayes (NB)**
- **XGBoost Classifier**
- **Neural Network**

---

### 🌳 Decision Tree
```python
from sklearn import tree
dtClf = tree.DecisionTreeClassifier(random_state=42, criterion='entropy', splitter='best')
dtClf.fit(X_train, y_train)
dt_y_pred = dtClf.predict(X_test)
```
**Accuracy:** 0.6488

---

### 🧪 Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB
NBClf = GaussianNB()
NBClf.fit(X_train, y_train)
NB_y_pred = NBClf.predict(X_test)
```
**Accuracy:** 0.6271

---

### 🌲 Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```
**Accuracy:** 0.7268

---

### ⚡ XGBoost
```python
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
le = LabelEncoder()
y_num = le.fit_transform(y)
X_train, X_test, y_num_train, y_num_test = train_test_split(X, y_num, test_size=0.4, random_state=42)
model = xgb.XGBClassifier(learning_rate=0.2, max_depth=100, n_estimators=100)
model.fit(X_train, y_num_train)
y_predict = model.predict(X_test)
```
**Accuracy:** 0.8650

---

### 🧠 Neural Network
```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)), BatchNormalization(), Dropout(0.3),
    Dense(64, activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(32, activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(y_train_cat.shape[1], activation='softmax')
])
```
**Test Accuracy:** 0.90

---

## 🏁 Conclusion

- All models confirm the **suitability** of the dataset for classification tasks.
- **XGBoost** and **Neural Networks** perform best with accuracy scores of **86.5%** and **90%** respectively.
- Extensive data cleaning helped remove outliers, noisy data, and redundant columns to improve model performance.
- The dataset is well-structured for further applications like **deep reinforcement learning** or **game move predictions**.

---
