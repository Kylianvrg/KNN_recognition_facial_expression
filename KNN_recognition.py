import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# LBP parameters (Radius and points)
rad = 8
points = 8 * rad

# Function to extract LBP features from the image
def lbp_features_extraction(img):
    lbp = local_binary_pattern(img, points, rad, method='uniform')
    hist, _ = np.histogram(lbp, bins=points + 3, range=(0, points + 2))
    hist = hist / (hist.sum() + 1e-6)  # Normalize histogram
    return hist

# Function to load dataset images and their labels
def load_dataset(data_dir):
    img_data = []
    label_data = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Emotion categories
    
    for emotion in emotions:
        emotion_path = os.path.join(data_dir, emotion)
        for filename in os.listdir(emotion_path):
            img_full_path = os.path.join(emotion_path, filename)
            img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (48, 48))
                img_resized = cv2.equalizeHist(img_resized)  # Apply histogram equalization
                features = lbp_features_extraction(img_resized)  # Extract LBP features
                img_data.append(features)
                label_data.append(emotion)
    
    return np.array(img_data), np.array(label_data)


# Loading training and testing data
train_x, train_y = load_dataset('train')
test_x, test_y = load_dataset('test')

# Standardizing the dataset
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Hyperparameter tuning using grid search for KNN
param_search = {'n_neighbors': [3, 5, 10, 15, 20, 25, 30, 35, 40, 50], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), param_search, cv=5)
knn_grid.fit(train_x, train_y)

# Output the best parameters chosen by grid search
print("Optimal parameters found: ", knn_grid.best_params_)

# Training KNN model with optimal parameters
best_params = knn_grid.best_params_
knn_model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
knn_model.fit(train_x, train_y)

# Model evaluation on the test set
test_predictions = knn_model.predict(test_x)
print("Classification Report:\n", classification_report(test_y, test_predictions))
print("Accuracy:", accuracy_score(test_y, test_predictions))


# Confusion matrix generation and visualization
conf_matrix = confusion_matrix(test_y, test_predictions)
plt.figure(figsize=(8, 6))
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Emotion categories
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
