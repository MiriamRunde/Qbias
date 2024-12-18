import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.utils import to_categorical

# --- Step 1: Load and Preprocess Data ---
# Load the dataset
file_path = "allsides_balanced_news_headlines-texts.csv"
data = pd.read_csv(file_path)

# Clean text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and digits
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    return ""

data['clean_text'] = data['text'].apply(clean_text)

# Encode bias_rating labels
label_mapping = {'left': 0, 'center': 1, 'right': 2}
data = data[data['bias_rating'].isin(label_mapping)]  # Filter valid bias ratings
y = data['bias_rating'].map(label_mapping).values

# --- Step 2: Tokenize and Pad Sequences ---
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

# Tokenize text
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(data['clean_text'])
sequences = tokenizer.texts_to_sequences(data['clean_text'])

# Pad sequences
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# One-hot encode labels
y = to_categorical(y, num_classes=3)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 3: Build the Neural Network Model ---
model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Output layer with 3 classes
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display the model summary
print("Model Summary:")
model.summary()

# --- Step 4: Train the Model ---
EPOCHS = 5
BATCH_SIZE = 32

print("\nTraining the model...")
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# --- Step 5: Evaluate the Model ---
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# --- Optional: Save the Model ---
model.save("bias_classifier_nn.h5")
print("Model saved as 'bias_classifier_nn.h5'")

s