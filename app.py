import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
st.set_page_config(page_title='Dementia Detection', layout='wide')
st.title('Dementia Detection from Language Features')
# Automatically converted from Kaggle notebook


import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())

    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

# st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['Label'], test_size=0.2, random_state=42)

# ✅ Train models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

# ✅ ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Evaluate ANN
y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")
st.write("\nANN Accuracy:", accuracy_score(y_test, y_pred_ann))
st.write(classification_report(y_test, y_pred_ann))

# ✅ Data Visualization
plt.figure(figsize=(8,5))
sns.countplot(x=df["Label"], palette="coolwarm")
plt.title("Class Distribution")
plt.xlabel("Label (0 = Control, 1 = Dementia)")
plt.ylabel("Count")
st.pyplot(plt.gcf())

plt.figure(figsize=(8,5))
sns.heatmap(pd.DataFrame(X_tfidf).corr(), cmap="coolwarm", linewidths=0.5)
plt.title("TF-IDF Feature Correlation")
st.pyplot(plt.gcf())



import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['Label'], test_size=0.2, random_state=42)

# ✅ Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
st.write("Class distribution after SMOTE:", pd.Series(y_train_balanced).value_counts())

# ✅ Train models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

# ✅ ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_balanced.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_balanced, y_train_balanced, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Evaluate ANN
y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")
st.write("\nANN Accuracy:", accuracy_score(y_test, y_pred_ann))
st.write(classification_report(y_test, y_pred_ann))

# ✅ Data Visualization
plt.figure(figsize=(8,5))
sns.countplot(x=df["Label"], palette="coolwarm")
plt.title("Class Distribution")
plt.xlabel("Label (0 = Control, 1 = Dementia)")
plt.ylabel("Count")
st.pyplot(plt.gcf())

plt.figure(figsize=(8,5))
sns.heatmap(pd.DataFrame(X_tfidf).corr(), cmap="coolwarm", linewidths=0.5)
plt.title("TF-IDF Feature Correlation")
st.pyplot(plt.gcf())



from collections import Counter
st.write("Before SMOTE:", Counter(y_train))
st.write("After SMOTE:", Counter(y_train_balanced))


import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Apply SMOTE before splitting the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['Label'])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# ✅ Verify class distribution after SMOTE
from collections import Counter
st.write("Before SMOTE:", Counter(df['Label']))
st.write("After SMOTE:", Counter(y_resampled))

# ✅ Train models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

# ✅ ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Evaluate ANN
y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")
st.write("\nANN Accuracy:", accuracy_score(y_test, y_pred_ann))
st.write(classification_report(y_test, y_pred_ann))

# ✅ Data Visualization
plt.figure(figsize=(8,5))
sns.countplot(x=pd.DataFrame({"Label": y_resampled})["Label"], palette="coolwarm")
plt.title("Class Distribution After SMOTE")
plt.xlabel("Label (0 = Control, 1 = Dementia)")
plt.ylabel("Count")
st.pyplot(plt.gcf())

plt.figure(figsize=(8,5))
sns.heatmap(pd.DataFrame(X_tfidf).corr(), cmap="coolwarm", linewidths=0.5)
plt.title("TF-IDF Feature Correlation")
st.pyplot(plt.gcf())


# ✅ Import libraries
import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE
from collections import Counter

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Visualize class distribution BEFORE and AFTER SMOTE vertically with counts
fig, axes = plt.subplots(2, 1, figsize=(8, 12))  # 2 rows, 1 column

# Before SMOTE
sns.countplot(x=df["Label"], palette="Set2", ax=axes[0])
axes[0].set_title("Class Distribution Before SMOTE", fontsize=14)
axes[0].set_xlabel("Label (0 = Control, 1 = Dementia)")
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                     textcoords='offset points')

# ✅ Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['Label'])

# After SMOTE
sns.countplot(x=pd.DataFrame({"Label": y_resampled})["Label"], palette="Set1", ax=axes[1])
axes[1].set_title("Class Distribution After SMOTE", fontsize=14)
axes[1].set_xlabel("Label (0 = Control, 1 = Dementia)")
axes[1].set_ylabel("Count")
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                     textcoords='offset points')

plt.tight_layout()
st.pyplot(plt.gcf())

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# ✅ Check class distribution
st.write("Before SMOTE:", Counter(df['Label']))
st.write("After SMOTE:", Counter(y_resampled))

# ✅ Train SVM, KNN, Random Forest models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

# ✅ ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Evaluate ANN
y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")
st.write("\nANN Accuracy:", accuracy_score(y_test, y_pred_ann))
st.write(classification_report(y_test, y_pred_ann))

# ✅ TF-IDF Feature Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(pd.DataFrame(X_tfidf).corr(), cmap="coolwarm", linewidths=0.5)
plt.title("TF-IDF Feature Correlation")
st.pyplot(plt.gcf())



import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE

# ✅ Set custom NLTK data path (for offline environments)
nltk.data.path.append("./nltk_data")

# ✅ Ensure NLTK resources are available
try:
    nltk.download('punkt', download_dir="./nltk_data")
    nltk.download('stopwords', download_dir="./nltk_data")
except Exception as e:
    st.write("NLTK download error:", e)

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Apply SMOTE before splitting the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['Label'])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# ✅ Verify class distribution after SMOTE
from collections import Counter
st.write("Before SMOTE:", Counter(df['Label']))
st.write("After SMOTE:", Counter(y_resampled))

# ✅ Train models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

# ✅ ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Evaluate ANN
y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")
st.write("\nANN Accuracy:", accuracy_score(y_test, y_pred_ann))
st.write(classification_report(y_test, y_pred_ann))

# ✅ Data Visualization
plt.figure(figsize=(8,5))
sns.countplot(x=pd.DataFrame({"Label": y_resampled})["Label"], palette="coolwarm")
plt.title("Class Distribution After SMOTE")
plt.xlabel("Label (0 = Control, 1 = Dementia)")
plt.ylabel("Count")
st.pyplot(plt.gcf())

plt.figure(figsize=(8,5))
sns.heatmap(pd.DataFrame(X_tfidf).corr(), cmap="coolwarm", linewidths=0.5)
plt.title("TF-IDF Feature Correlation")
st.pyplot(plt.gcf())












import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE
import scikitplot as skplt

# ✅ Set custom NLTK data path (for offline environments)
nltk.data.path.append("./nltk_data")

# ✅ Ensure NLTK resources are available
try:
    nltk.download('punkt', download_dir="./nltk_data")
    nltk.download('stopwords', download_dir="./nltk_data")
except Exception as e:
    st.write("NLTK download error:", e)

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Apply SMOTE before splitting the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['Label'])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# ✅ Train models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))
    
    # ✅ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'Dementia'], yticklabels=['Control', 'Dementia'])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt.gcf())
    
    # ✅ ROC Curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())

# ✅ Class Distribution Visualization
plt.figure(figsize=(8,5))
sns.countplot(x=pd.DataFrame({"Label": y_resampled})["Label"], palette="coolwarm")
plt.title("Class Distribution After SMOTE")
plt.xlabel("Label (0 = Control, 1 = Dementia)")
plt.ylabel("Count")
st.pyplot(plt.gcf())

plt.figure(figsize=(8,5))
sns.heatmap(pd.DataFrame(X_tfidf).corr(), cmap="coolwarm", linewidths=0.5)
plt.title("TF-IDF Feature Correlation")
st.pyplot(plt.gcf())

import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()

# ✅ Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_tfidf, y_resampled = smote.fit_resample(X_tfidf, df['Label'])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_resampled, test_size=0.2, random_state=42)

# ✅ Train models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))
    
    # ✅ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt.gcf())
    
    # ✅ ROC Curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} - ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())

# ✅ ANN model
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Evaluate ANN
y_pred_ann = (ann_model.predict(X_test) > 0.5).astype("int32")
st.write("\nANN Accuracy:", accuracy_score(y_test, y_pred_ann))
st.write(classification_report(y_test, y_pred_ann))



import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE
from boruta import BorutaPy
import scikitplot as skplt

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())

    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Apply Boruta Feature Selection
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
boruta.fit(X_tfidf, df['Label'].values)
X_selected = boruta.transform(X_tfidf)
st.write("Boruta feature selection completed! Shape:", X_selected.shape)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, df['Label'], test_size=0.2, random_state=42)

# ✅ Train models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

# ✅ ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Evaluate ANN
y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")
st.write("\nANN Accuracy:", accuracy_score(y_test, y_pred_ann))
st.write(classification_report(y_test, y_pred_ann))

# ✅ Data Visualization
plt.figure(figsize=(8,5))
sns.countplot(x=df["Label"], palette="coolwarm")
plt.title("Class Distribution")
plt.xlabel("Label (0 = Control, 1 = Dementia)")
plt.ylabel("Count")
st.pyplot(plt.gcf())

# ✅ Confusion Matrix for Random Forest
y_pred_rf = models["Random Forest"].predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, y_pred_rf)
st.pyplot(plt.gcf())

# ✅ ROC Curve for Random Forest
y_probas_rf = models["Random Forest"].predict_proba(X_test)
skplt.metrics.plot_roc(y_test, y_probas_rf)
st.pyplot(plt.gcf())



import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())

    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['Label'])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Train models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

    # ✅ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    st.pyplot(plt.gcf())

    # ✅ ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())

# ✅ ANN model
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Evaluate ANN
y_pred_ann = (ann_model.predict(X_test) > 0.5).astype("int32")
st.write("\nANN Accuracy:", accuracy_score(y_test, y_pred_ann))
st.write(classification_report(y_test, y_pred_ann))



import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import scikitplot as skplt

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())

    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_tfidf, y = smote.fit_resample(X_tfidf, df['Label'])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# ✅ Define Bayesian Optimization functions
def optimize_svm(C):
    model = SVC(kernel='linear', C=C, probability=True)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def optimize_rf(n_estimators, max_depth):
    model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def optimize_lr(C):
    model = LogisticRegression(C=C, solver='liblinear')
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

# ✅ Perform Bayesian Optimization
svm_bo = BayesianOptimization(f=optimize_svm, pbounds={'C': (0.1, 10)}, random_state=42)
svm_bo.maximize()

rf_bo = BayesianOptimization(f=optimize_rf, pbounds={'n_estimators': (50, 200), 'max_depth': (5, 50)}, random_state=42)
rf_bo.maximize()

lr_bo = BayesianOptimization(f=optimize_lr, pbounds={'C': (0.1, 10)}, random_state=42)
lr_bo.maximize()

# ✅ Train models with optimized hyperparameters
best_svm = SVC(kernel='linear', C=svm_bo.max['params']['C'], probability=True)
best_rf = RandomForestClassifier(n_estimators=int(rf_bo.max['params']['n_estimators']), max_depth=int(rf_bo.max['params']['max_depth']), random_state=42)
best_lr = LogisticRegression(C=lr_bo.max['params']['C'], solver='liblinear')

models = {"SVM": best_svm, "Random Forest": best_rf, "Logistic Regression": best_lr}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

    # ✅ Confusion Matrix
    plt.figure(figsize=(5,4))
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    plt.title(f"Confusion Matrix - {name}")
    st.pyplot(plt.gcf())

    # ✅ ROC Curve
    y_probas = model.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y_probas)
    plt.title(f"ROC Curve - {name}")
    st.pyplot(plt.gcf())

st.write("Model Training and Evaluation Complete!")

import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import scikitplot as skplt

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path (NO ZIP FILE)
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files in subdirectories
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

st.write("Found .cha files:", cha_files)
if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())

    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset from .cha files
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("Error: No valid text data found after extraction. Check .cha file parsing.")

st.write("Sample data:\n", df.head())

# ✅ Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"]
df = df.dropna(subset=["Processed_Text"])

st.write("Sample processed text before TF-IDF:\n", df["Processed_Text"].head())

# ✅ Convert text into numerical representation using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()
st.write("TF-IDF transformation successful! Shape:", X_tfidf.shape)

# ✅ Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_tfidf, y = smote.fit_resample(X_tfidf, df['Label'])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# ✅ Define Bayesian Optimization functions
def optimize_svm(C):
    model = SVC(kernel='linear', C=C, probability=True)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def optimize_rf(n_estimators, max_depth):
    model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def optimize_lr(C):
    model = LogisticRegression(C=C, solver='liblinear')
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

# ✅ Perform Bayesian Optimization
svm_bo = BayesianOptimization(f=optimize_svm, pbounds={'C': (0.1, 10)}, random_state=42)
svm_bo.maximize()

rf_bo = BayesianOptimization(f=optimize_rf, pbounds={'n_estimators': (50, 200), 'max_depth': (5, 50)}, random_state=42)
rf_bo.maximize()

lr_bo = BayesianOptimization(f=optimize_lr, pbounds={'C': (0.1, 10)}, random_state=42)
lr_bo.maximize()

# ✅ Train models with optimized hyperparameters
best_svm = SVC(kernel='linear', C=svm_bo.max['params']['C'], probability=True)
best_rf = RandomForestClassifier(n_estimators=int(rf_bo.max['params']['n_estimators']), max_depth=int(rf_bo.max['params']['max_depth']), random_state=42)
best_lr = LogisticRegression(C=lr_bo.max['params']['C'], solver='liblinear')

models = {"SVM": best_svm, "Random Forest": best_rf, "Logistic Regression": best_lr}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

    # ✅ Confusion Matrix
    plt.figure(figsize=(5,4))
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    plt.title(f"Confusion Matrix - {name}")
    st.pyplot(plt.gcf())

    # ✅ ROC Curve
    y_probas = model.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y_probas)
    plt.title(f"ROC Curve - {name}")
    st.pyplot(plt.gcf())

st.write("Model Training and Evaluation Complete!")



import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("No valid text data found after extraction.")

# ✅ Text Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"].dropna()

# ✅ TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Text']).toarray()

# ✅ Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['Label'])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Bayesian Optimization
def optimize_svm(C):
    model = SVC(C=C, kernel='linear', probability=True)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

svm_bo = BayesianOptimization(f=optimize_svm, pbounds={"C": (0.1, 10)}, random_state=42)
svm_bo.maximize()
best_C = svm_bo.max["params"]["C"]

# ✅ Train models with best hyperparameters
models = {
    "SVM": SVC(kernel='linear', C=best_C, probability=True),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

    # ✅ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    st.pyplot(plt.gcf())

    # ✅ ROC Curve
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    st.pyplot(plt.gcf())



import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("No valid text data found after extraction.")

# ✅ Text Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"].dropna()

# ✅ Train Word2Vec Model
word2vec_model = Word2Vec(sentences=df["Processed_Text"], vector_size=100, window=5, min_count=1, workers=4)

# ✅ Convert Sentences into Feature Vectors
def sentence_vector(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

X_word2vec = np.array([sentence_vector(tokens, word2vec_model, 100) for tokens in df["Processed_Text"]])

# ✅ Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_word2vec)

# ✅ Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, df["Label"])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Train Models with Word2Vec Features
models = {
    "SVM": SVC(kernel='linear', C=1.0, probability=True),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# ✅ Train & Evaluate Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

    # ✅ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    st.pyplot(plt.gcf())

    # ✅ ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        st.pyplot(plt.gcf())



import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ✅ Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata annotations
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Remove punctuation but keep colons
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("No valid text data found after extraction.")

# ✅ Text Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"].dropna()

# ✅ Train Word2Vec Model
word2vec_model = Word2Vec(sentences=[text.split() for text in df["Processed_Text"]], vector_size=100, window=5, min_count=1, workers=4)

# ✅ Convert Sentences into Feature Vectors
def sentence_vector(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

X_word2vec = np.array([sentence_vector(text.split(), word2vec_model, 100) for text in df["Processed_Text"]])

# ✅ Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_word2vec)

# ✅ Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, df["Label"])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Bayesian Optimization for SVM, Random Forest, and Decision Tree
def optimize_svm(C):
    model = SVC(C=C, kernel='linear', probability=True)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

svm_bo = BayesianOptimization(f=optimize_svm, pbounds={"C": (0.1, 10)}, random_state=42)
svm_bo.maximize()
best_C = svm_bo.max["params"]["C"]

def optimize_rf(n_estimators):
    model = RandomForestClassifier(n_estimators=int(n_estimators), random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

rf_bo = BayesianOptimization(f=optimize_rf, pbounds={"n_estimators": (50, 200)}, random_state=42)
rf_bo.maximize()
best_n_estimators = int(rf_bo.max["params"]["n_estimators"])

def optimize_dt(max_depth):
    model = DecisionTreeClassifier(max_depth=int(max_depth), random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

dt_bo = BayesianOptimization(f=optimize_dt, pbounds={"max_depth": (3, 20)}, random_state=42)
dt_bo.maximize()
best_max_depth = int(dt_bo.max["params"]["max_depth"])

# ✅ Train Models with Optimized Hyperparameters
models = {
    "SVM": SVC(kernel='linear', C=best_C, probability=True),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=best_n_estimators, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
}

# ✅ Train & Evaluate Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

    # ✅ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    st.pyplot(plt.gcf())

    # ✅ ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        st.pyplot(plt.gcf())



import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# ✅ Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Define dataset path
dataset_path = "/kaggle/input/major-project"

# ✅ Search for .cha files
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ Function to extract text from .cha files
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)  # Remove metadata
                line = re.sub(r"[^a-zA-Z\s:]", "", line)  # Keep only text
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# ✅ Load dataset
data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("No valid text data found after extraction.")

# ✅ Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"].dropna()

# ✅ Tokenize for embedding
df["Tokenized_Text"] = df["Processed_Text"].apply(word_tokenize)

# ✅ Load GloVe embeddings
glove_input_file = "/kaggle/input/glove-dataset/glove.6B.100d.txt"  # Place this file in working dir
word2vec_output_file = "glove.6B.100d.word2vec.txt"

if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)

glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# ✅ Convert sentences to GloVe vectors
def glove_sentence_vector(tokens, model, vector_size):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

X_glove = np.array([glove_sentence_vector(tokens, glove_model, 100) for tokens in df["Tokenized_Text"]])

# ✅ Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_glove)

# ✅ Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, df["Label"])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Define models
models = {
    "SVM": SVC(kernel='linear', C=1.0, probability=True),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# ✅ Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    st.pyplot(plt.gcf())

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        st.pyplot(plt.gcf())



# ✅ IMPORTS
import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from bayes_opt import BayesianOptimization

# ✅ DOWNLOAD NLTK DATA
nltk.download('punkt')
nltk.download('stopwords')

# ✅ LOAD DATASET
dataset_path = "/kaggle/input/major-project"
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ EXTRACT TEXT FROM CHA
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)
                line = re.sub(r"[^a-zA-Z\s:]", "", line)
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("No valid text data found after extraction.")

# ✅ TEXT PREPROCESSING
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"].dropna()

# ✅ GLOVE EMBEDDING CONVERSION & LOADING
glove_input_file = "/kaggle/input/glove-dataset/glove.6B.100d.txt"
word2vec_output_file = "glove.6B.100d.word2vec.txt"

if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)

glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# ✅ CONVERT SENTENCE TO VECTOR
def sentence_vector(text, model, vector_size):
    tokens = text.split()
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

X_glove = np.array([sentence_vector(text, glove_model, 100) for text in df["Processed_Text"]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_glove)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, df["Label"])
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# -------------------------------
# BAYESIAN OPTIMIZATION SECTION
# -------------------------------

# --- Logistic Regression Optimization ---
def optimize_logistic(C):
    model = LogisticRegression(C=C, solver='liblinear', max_iter=300)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

log_bounds = {'C': (0.001, 10.0)}
log_optimizer = BayesianOptimization(f=optimize_logistic, pbounds=log_bounds, random_state=42)
log_optimizer.maximize(init_points=5, n_iter=10)
best_C = log_optimizer.max['params']['C']
log_model = LogisticRegression(C=best_C, solver='liblinear', max_iter=300)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

st.write("\n🔵 Logistic Regression (Optimized)")
st.write(f"Best C: {best_C:.4f}")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
st.write(classification_report(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression (Optimized)")
st.pyplot(plt.gcf())

fpr, tpr, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc(fpr, tpr):.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (Optimized)")
plt.legend()
st.pyplot(plt.gcf())

# --- Random Forest Optimization ---
def optimize_rf(n_estimators, max_depth, min_samples_split):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        random_state=42
    )
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

rf_bounds = {
    'n_estimators': (50, 200),
    'max_depth': (5, 30),
    'min_samples_split': (2, 10)
}
rf_optimizer = BayesianOptimization(f=optimize_rf, pbounds=rf_bounds, random_state=42)
rf_optimizer.maximize(init_points=5, n_iter=10)
rf_params = rf_optimizer.max['params']
rf_model = RandomForestClassifier(
    n_estimators=int(rf_params['n_estimators']),
    max_depth=int(rf_params['max_depth']),
    min_samples_split=int(rf_params['min_samples_split']),
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

st.write("\n🟢 Random Forest (Optimized)")
st.write(f"Best Params: {rf_params}")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
st.write(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest (Optimized)")
st.pyplot(plt.gcf())

fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc(fpr, tpr):.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest (Optimized)")
plt.legend()
st.pyplot(plt.gcf())

# --- SVM Optimization ---
def optimize_svm(C, gamma):
    model = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

svm_bounds = {'C': (0.1, 10), 'gamma': (1e-4, 1e-1)}
svm_optimizer = BayesianOptimization(f=optimize_svm, pbounds=svm_bounds, random_state=42)
svm_optimizer.maximize(init_points=5, n_iter=10)
svm_params = svm_optimizer.max['params']
svm_model = SVC(C=svm_params['C'], gamma=svm_params['gamma'], kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

st.write("\n🟣 SVM (Optimized)")
st.write(f"Best Params: {svm_params}")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
st.write(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM (Optimized)")
st.pyplot(plt.gcf())

fpr, tpr, _ = roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"SVM (AUC = {auc(fpr, tpr):.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM (Optimized)")
plt.legend()
st.pyplot(plt.gcf())

# -------------------------------
# ENSEMBLE: VOTING CLASSIFIER
# -------------------------------
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('lr', log_model),
        ('rf', rf_model)
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
y_pred_ensemble = voting_clf.predict(X_test)
y_proba_ensemble = voting_clf.predict_proba(X_test)[:, 1]

st.write("\n✅ Ensemble Voting Classifier Results:")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
st.write(classification_report(y_test, y_pred_ensemble))

cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ensemble Voting Classifier")
st.pyplot(plt.gcf())

fpr, tpr, _ = roc_curve(y_test, y_proba_ensemble)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Ensemble (AUC = {auc(fpr, tpr):.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Ensemble Voting Classifier")
plt.legend()
st.pyplot(plt.gcf())

# ✅ IMPORTS
import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# ✅ DOWNLOAD NLTK DATA
nltk.download('punkt')
nltk.download('stopwords')

# ✅ LOAD DATASET
dataset_path = "/kaggle/input/major-project"
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ EXTRACT TEXT FROM CHA
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)
                line = re.sub(r"[^a-zA-Z\s:]", "", line)
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("No valid text data found after extraction.")

# ✅ TEXT PREPROCESSING
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"].dropna()

# ✅ LOAD SENTENCE TRANSFORMER FROM LOCAL PATH
local_model_path = "/kaggle/input/model1/my_local_model"
sentence_model = SentenceTransformer(local_model_path)
X_embeddings = sentence_model.encode(df["Processed_Text"].tolist(), show_progress_bar=True)

# ✅ STANDARDIZATION + SMOTE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_embeddings)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, df["Label"])

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ BAYESIAN OPTIMIZATION FOR SELECT MODELS
search_spaces = {
    "SVM": {
        "C": Real(1e-3, 100.0, prior='log-uniform'),
        "kernel": ["linear", "rbf"],
        "gamma": Real(1e-4, 1.0, prior='log-uniform')
    },
    "Logistic Regression": {
        "C": Real(1e-3, 100.0, prior='log-uniform'),
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"]
    },
    "Random Forest": {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 30),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 10)
    }
}

base_models = {
    "SVM": SVC(probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

optimized_models = {}
for name, model in base_models.items():
    st.write(f"\n🔍 Tuning {name} with Bayesian Optimization...")
    opt = BayesSearchCV(
        model,
        search_spaces[name],
        n_iter=20,
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    opt.fit(X_train, y_train)
    best_model = opt.best_estimator_
    optimized_models[name] = best_model

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    st.write(f"\n✅ {name} (Bayes Optimized) Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name} (Optimized)")
    st.pyplot(plt.gcf())

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name} (Optimized)")
    plt.legend()
    st.pyplot(plt.gcf())

# ✅ NON-OPTIMIZED MODELS
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    st.write(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}") 
    st.pyplot(plt.gcf())

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        st.pyplot(plt.gcf())



# ✅ IMPORTS
import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# ✅ DOWNLOAD NLTK DATA
nltk.download('punkt')
nltk.download('stopwords')

# ✅ LOAD DATASET
dataset_path = "/kaggle/input/major-project"
cha_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".cha"):
            cha_files.append(os.path.join(root, file))

if not cha_files:
    raise FileNotFoundError("No .cha files found! Check dataset structure.")

# ✅ EXTRACT TEXT FROM CHA
def extract_text_from_cha(file_path):
    extracted_text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("*PAR:") or line.startswith("*CHI:") or line.startswith("*SPE:"):
                line = re.sub(r"\[.*?\]", "", line)
                line = re.sub(r"[^a-zA-Z\s:]", "", line)
                if ":" in line:
                    extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

data, labels = [], []
for file_path in cha_files:
    text = extract_text_from_cha(file_path)
    label = 0 if "control" in file_path.lower() else 1 if "dementia" in file_path.lower() else None
    if label is not None:
        data.append(text)
        labels.append(label)

df = pd.DataFrame({"Text": data, "Label": labels})
df = df[df["Text"] != "empty_transcription"]

if df.empty:
    raise ValueError("No valid text data found after extraction.")

# ✅ TEXT PREPROCESSING
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens) if tokens else "meaningful_word"

df["Processed_Text"] = df["Text"].apply(preprocess_text)
df = df[df["Processed_Text"] != "meaningful_word"].dropna()

# ✅ LOAD SENTENCE TRANSFORMER FROM LOCAL PATH
local_model_path = "/kaggle/input/model1/my_local_model"
sentence_model = SentenceTransformer(local_model_path)
X_embeddings = sentence_model.encode(df["Processed_Text"].tolist(), show_progress_bar=True)

# ✅ STANDARDIZATION + SMOTE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_embeddings)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, df["Label"])

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ BAYESIAN OPTIMIZATION
search_spaces = {
    "SVM": {
        "C": Real(1e-3, 100.0, prior='log-uniform'),
        "kernel": ["linear", "rbf"],
        "gamma": Real(1e-4, 1.0, prior='log-uniform')
    },
    "Logistic Regression": {
        "C": Real(1e-3, 100.0, prior='log-uniform'),
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"]
    },
    "Random Forest": {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 30),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 10)
    }
}

base_models = {
    "SVM": SVC(probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

optimized_models = {}
for name, model in base_models.items():
    st.write(f"\n🔍 Tuning {name} with Bayesian Optimization...")
    opt = BayesSearchCV(
        model,
        search_spaces[name],
        n_iter=20,
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    opt.fit(X_train, y_train)
    best_model = opt.best_estimator_
    optimized_models[name] = best_model

# ✅ ENSEMBLE: VOTING CLASSIFIER (Soft Voting)
voting_clf = VotingClassifier(
    estimators=[
        ('svm', optimized_models["SVM"]),
        ('lr', optimized_models["Logistic Regression"]),
        ('rf', optimized_models["Random Forest"]),
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
y_proba = voting_clf.predict_proba(X_test)[:, 1]

st.write("\n✅ Ensemble Voting Classifier Results:")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
st.write(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ensemble Voting Classifier")
st.pyplot(plt.gcf())

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Ensemble (AUC = {auc(fpr, tpr):.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Ensemble Voting Classifier")
plt.legend()
st.pyplot(plt.gcf())


