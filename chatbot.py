from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Define training questions and their corresponding answers/labels
training_questions = [
    "What is the netflix login and password?",
    "How do I log into Netflix?",
    "What is the Netflix account info?",
    
    "What is the wifi password?",
    "How do I connect to the Wi-Fi?",
    "What is the internet network name?",
    
    "When is trash day?",
    "What day does the garbage go out?",
    "When do we take out the trash?"
]

# Labels mapping directly to the response you want
answers = [
    "Netflix -> Email: family@gmail.com | Pass: 12345",
    "Netflix -> Email: family@gmail.com | Pass: 12345",
    "Netflix -> Email: family@gmail.com | Pass: 12345",
    
    "Wi-Fi -> Network: OurHomeWifi | Pass: SuperSecret123",
    "Wi-Fi -> Network: OurHomeWifi | Pass: SuperSecret123",
    "Wi-Fi -> Network: OurHomeWifi | Pass: SuperSecret123",
    
    "Trash -> Pickup is every Tuesday morning!",
    "Trash -> Pickup is every Tuesday morning!",
    "Trash -> Pickup is every Tuesday morning!"
]

# Step 2: Convert text questions into numerical vectors (TF-IDF)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_questions)

# Step 3: Train the KNN Classifier
# n_neighbors=1 finds the single closest matching question
knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
knn.fit(X_train, answers)

# Step 4: Query Function
def ask_knn_bot(user_input):
    # Vectorize input text using the same vectorizer
    input_vector = vectorizer.transform([user_input])
    
    # Predict closest matching answer
    prediction = knn.predict(input_vector)
    return prediction[0]

# --- Testing ---
print(ask_knn_bot("can I get the netflix password"))
# Output: Netflix -> Email: family@gmail.com | Pass: 12345

print(ask_knn_bot("wifi network name"))
# Output: Wi-Fi -> Network: OurHomeWifi | Pass: SuperSecret123