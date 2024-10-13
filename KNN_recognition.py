import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# LBP parameters
radius = 16
n_points = 8 * radius

# Function to extract LBP features from an image
def extract_lbp_features(image):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=n_points + 3, range=(0, n_points + 2))
    hist = hist / (hist.sum() + 1e-6)  # Normalization
    return hist

# Function to load images and labels from the dataset
def load_data(directory):
    images = []
    labels = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Emotion categories
    
    for emotion in emotions:
        emotion_dir = os.path.join(directory, emotion)
        for file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (48, 48))
                image = cv2.equalizeHist(image)  # Histogram equalization
                features = extract_lbp_features(image)  # Extract LBP features
                images.append(features)
                labels.append(emotion)
    
    return np.array(images), np.array(labels)

# Load training and testing data
train_data, train_labels = load_data('train')
test_data, test_labels = load_data('test')

# Standardize the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Apply PCA for dimensionality reduction
n_components = min(50, train_data.shape[1])  # Adjust number of components based on input size
pca = PCA(n_components=n_components)
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)

# Grid search to find the best hyperparameters for KNN
param_grid = {'n_neighbors': [3, 5, 10, 15, 20, 25, 30, 35, 40], 'weights': ['uniform', 'distance']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(train_data, train_labels)

# Output the best parameters
print("Best parameters found: ", grid.best_params_)

# Train the KNN model using the best parameters
best_params = grid.best_params_
knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
knn.fit(train_data, train_labels)

# Evaluate the model on test data
predictions = knn.predict(test_data)
print("Classification Report:\n", classification_report(test_labels, predictions))
print("Accuracy:", accuracy_score(test_labels, predictions))

# Confusion matrix creation and visualization
conf_matrix = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], yticklabels=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
