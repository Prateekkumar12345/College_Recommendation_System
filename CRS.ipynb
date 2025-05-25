import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import sentence transformers, fallback to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ SentenceTransformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ SentenceTransformers not available, using TF-IDF fallback")

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ All libraries imported successfully!")

class TextEmbedder:
    """
    A wrapper class that uses SentenceTransformers if available,
    otherwise falls back to TF-IDF vectorization
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.use_sentence_transformers = False
        self.model = None
        self.tfidf = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Try to load with offline mode first
                self.model = SentenceTransformer(model_name, device='cpu')
                self.use_sentence_transformers = True
                print(f"✅ Using SentenceTransformer: {model_name}")
            except Exception as e:
                print(f"⚠️ SentenceTransformer failed: {e}")
                print("📝 Falling back to TF-IDF vectorization")
                self._setup_tfidf()
        else:
            print("📝 Using TF-IDF vectorization")
            self._setup_tfidf()

    def _setup_tfidf(self):
        """Setup TF-IDF vectorizer as fallback"""
        self.tfidf = TfidfVectorizer(
            max_features=384,  # Similar to sentence transformer dimension
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.use_sentence_transformers = False

    def encode(self, texts):
        """
        Encode texts using either SentenceTransformer or TF-IDF

        Parameters:
        -----------
        texts : list
            List of texts to encode

        Returns:
        --------
        numpy.ndarray
            Encoded vectors
        """
        if self.use_sentence_transformers and self.model is not None:
            try:
                return self.model.encode(texts, show_progress_bar=False)
            except Exception as e:
                print(f"⚠️ SentenceTransformer encoding failed: {e}")
                print("📝 Switching to TF-IDF fallback")
                self._setup_tfidf()
                return self._encode_with_tfidf(texts)
        else:
            return self._encode_with_tfidf(texts)

    def _encode_with_tfidf(self, texts):
        """Encode texts using TF-IDF"""
        if not hasattr(self.tfidf, 'vocabulary_') or self.tfidf.vocabulary_ is None:
            # First time - fit the vectorizer
            vectors = self.tfidf.fit_transform(texts)
        else:
            # Already fitted - just transform
            vectors = self.tfidf.transform(texts)

        return vectors.toarray()

class StudentModel:
    def __init__(self, num_clusters=5, random_state=42):
        """
        Initialize the StudentModel for processing student data

        Parameters:
        -----------
        num_clusters : int
            Number of clusters to group students into
        random_state : int
            Random seed for reproducibility
        """
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.text_embedder = TextEmbedder('all-MiniLM-L6-v2')
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.pca = PCA(n_components=10)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        self.numerical_columns = ['Marks_10th', 'Marks_12th', 'JEE_Score', 'Budget']
        self.categorical_columns = [
            'Preferred Location', 'Gender', 'Target Exam', 'State Board', 'Category',
            'Stress Tolerance', 'English Proficiency'
        ]
        self.text_columns = ['Extra Curriculars', 'Future Goal', 'Certifications']
        self.fitted = False

    def preprocess_data(self, df):
        """
        Preprocess the student data by scaling numerical features and encoding categorical features

        Parameters:
        -----------
        df : pandas.DataFrame
            Student dataset

        Returns:
        --------
        tuple
            Processed numerical, categorical, and text features
        """
        # Handle numerical features
        numerical_features = df[self.numerical_columns].copy()

        # Handle categorical features
        categorical_features = df[self.categorical_columns].copy()

        # Handle text features - combine for embedding
        text_features = []
        for _, row in df.iterrows():
            text = f"{row['Extra Curriculars']} {row['Certifications']} {row['Future Goal']}"
            text_features.append(text)

        return numerical_features, categorical_features, text_features

    def fit(self, df):
        """
        Fit the model to the student data

        Parameters:
        -----------
        df : pandas.DataFrame
            Student dataset

        Returns:
        --------
        self
        """
        print("🔄 Processing student data...")

        # Preprocess data
        numerical_features, categorical_features, text_features = self.preprocess_data(df)

        # Scale numerical features
        scaled_numerical = self.numerical_scaler.fit_transform(numerical_features)
        print(f"✅ Scaled {scaled_numerical.shape[1]} numerical features")

        # Encode categorical features
        encoded_categorical = self.categorical_encoder.fit_transform(categorical_features)
        print(f"✅ Encoded {encoded_categorical.shape[1]} categorical features")

        # Embed text features
        print("🔄 Encoding text features...")
        text_embeddings = self.text_embedder.encode(text_features)
        print(f"✅ Generated {text_embeddings.shape[1]} text embedding features")

        # Combine all features
        combined_features = np.hstack([scaled_numerical, encoded_categorical, text_embeddings])
        print(f"✅ Combined features shape: {combined_features.shape}")

        # Apply PCA for dimensionality reduction
        self.pca_features = self.pca.fit_transform(combined_features)
        print(f"✅ PCA reduced to {self.pca_features.shape[1]} components")

        # Apply KMeans clustering
        self.cluster_labels = self.kmeans.fit_predict(self.pca_features)
        print(f"✅ KMeans clustering completed")

        # Store original data
        self.original_data = df.copy()

        # Add cluster labels to original data
        self.original_data['Cluster'] = self.cluster_labels

        self.fitted = True
        return self

    def predict_cluster(self, student_dict):
        """
        Predict the cluster for a new student

        Parameters:
        -----------
        student_dict : dict
            Dictionary containing student information

        Returns:
        --------
        tuple
            Predicted cluster ID and PCA features
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Convert dictionary to DataFrame
        student_df = pd.DataFrame([student_dict])

        # Preprocess the new student data
        numerical_features, categorical_features, text_features = self.preprocess_data(student_df)

        # Scale numerical features
        scaled_numerical = self.numerical_scaler.transform(numerical_features)

        # Encode categorical features
        encoded_categorical = self.categorical_encoder.transform(categorical_features)

        # Embed text features
        text_embeddings = self.text_embedder.encode(text_features)

        # Combine all features
        combined_features = np.hstack([scaled_numerical, encoded_categorical, text_embeddings])

        # Apply PCA transformation
        pca_features = self.pca.transform(combined_features)

        # Predict cluster
        cluster = self.kmeans.predict(pca_features)[0]

        return cluster, pca_features

    def get_students_in_cluster(self, cluster_id):
        """
        Get all students in a specific cluster

        Parameters:
        -----------
        cluster_id : int
            Cluster ID

        Returns:
        --------
        pandas.DataFrame
            Students in the specified cluster
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.original_data[self.original_data['Cluster'] == cluster_id]

    def visualize_clusters(self):
        """
        Visualize the student clusters in 2D PCA space
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Use first two PCA components for visualization
        pca_2d = PCA(n_components=2)
        pca_result_2d = pca_2d.fit_transform(self.pca_features)

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=pca_result_2d[:, 0], y=pca_result_2d[:, 1], hue=self.cluster_labels, palette='viridis')
        plt.title('Student Clusters in 2D PCA Space')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend(title='Cluster')
        plt.show()

print("✅ StudentModel class defined successfully!")

class CollegeModel:
    def __init__(self, num_clusters=8, random_state=42):
        """
        Initialize the CollegeModel for processing college data

        Parameters:
        -----------
        num_clusters : int
            Number of clusters to group colleges into
        random_state : int
            Random seed for reproducibility
        """
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.text_embedder = TextEmbedder('all-MiniLM-L6-v2')
        self.numerical_scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        self.fitted = False

    def preprocess_data(self, df):
        """
        Preprocess the college data by creating a textual representation and extracting numerical features

        Parameters:
        -----------
        df : pandas.DataFrame
            College dataset

        Returns:
        --------
        tuple
            Processed text features and numerical features
        """
        # Extract important numerical features
        numerical_features = []

        # Extract rankings and convert to numeric
        nirf_ranking = pd.to_numeric(df['NIRF Ranking'], errors='coerce')
        numerical_features.append(nirf_ranking)

        # Extract and process fees (assuming it's in LPA format)
        fees_data = []
        for fee in df['Course Fees (₹)']:
            if isinstance(fee, str) and 'LPA' in fee:
                try:
                    # Extract numeric part before LPA
                    value = float(fee.replace('LPA', '').replace('₹', '').strip())
                    fees_data.append(value)
                except:
                    fees_data.append(np.nan)
            else:
                fees_data.append(np.nan)

        numerical_features.append(fees_data)

        # Convert to numpy array and transpose
        numerical_features = np.array(numerical_features).T

        # Replace NaN values with column means
        for col in range(numerical_features.shape[1]):
            col_mean = np.nanmean(numerical_features[:, col])
            numerical_features[:, col] = np.nan_to_num(numerical_features[:, col], nan=col_mean)

        # Create textual representation for each college
        text_features = []
        for _, row in df.iterrows():
            college_text = f"College {row['College Name']} located in {row['Location']}, {row['State']} "
            college_text += f"is a {row['College Type']} established in {row['Established Year']}. "
            college_text += f"It is approved by {row['Approved By']} with NIRF ranking {row['NIRF Ranking']}. "
            college_text += f"The college offers courses in {row['Notable Courses Offered']}. "

            # Add information about education loan and placement
            if 'Education Loan' in row and row['Education Loan'] == 'Yes':
                college_text += "Education loan facility is available. "

            if 'Placement (Average' in row and not pd.isna(row['Placement (Average']):
                college_text += f"Average placement is {row['Placement (Average']}. "

            text_features.append(college_text)

        return text_features, numerical_features

    def fit(self, df):
        """
        Fit the model to the college data

        Parameters:
        -----------
        df : pandas.DataFrame
            College dataset

        Returns:
        --------
        self
        """
        print("🔄 Processing college data...")

        # Preprocess data
        text_features, numerical_features = self.preprocess_data(df)

        # Scale numerical features
        scaled_numerical = self.numerical_scaler.fit_transform(numerical_features)
        print(f"✅ Scaled {scaled_numerical.shape[1]} numerical features")

        # Embed text features
        print("🔄 Encoding college text features...")
        text_embeddings = self.text_embedder.encode(text_features)
        print(f"✅ Generated {text_embeddings.shape[1]} text embedding features")

        # Combine all features
        combined_features = np.hstack([scaled_numerical, text_embeddings])
        print(f"✅ Combined features shape: {combined_features.shape}")

        # Check for NaN values before PCA
        if np.isnan(combined_features).any():
            print("⚠️ Warning: NaN values found in combined features, replacing with zeros")
            combined_features = np.nan_to_num(combined_features)

        # Apply PCA for dimensionality reduction
        self.pca_features = self.pca.fit_transform(combined_features)
        print(f"✅ PCA reduced to {self.pca_features.shape[1]} components")

        # Apply KMeans clustering
        self.cluster_labels = self.kmeans.fit_predict(self.pca_features)
        print(f"✅ KMeans clustering completed")

        # Store original data
        self.original_data = df.copy()

        # Add cluster labels and PCA features to original data
        self.original_data['Cluster'] = self.cluster_labels
        for i in range(min(5, self.pca_features.shape[1])):  # Store first 5 PCA features
            self.original_data[f'PCA_{i+1}'] = self.pca_features[:, i]

        self.fitted = True
        return self

    def get_colleges_in_cluster(self, cluster_id):
        """
        Get all colleges in a specific cluster

        Parameters:
        -----------
        cluster_id : int
            Cluster ID

        Returns:
        --------
        pandas.DataFrame
            Colleges in the specified cluster
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.original_data[self.original_data['Cluster'] == cluster_id]

    def get_pca_features(self):
        """
        Get the PCA features for all colleges

        Returns:
        --------
        numpy.ndarray
            PCA features
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.pca_features

    def visualize_clusters(self):
        """
        Visualize the college clusters in 2D PCA space
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Use first two PCA components for visualization
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=self.pca_features[:, 0],
            y=self.pca_features[:, 1],
            hue=self.cluster_labels,
            palette='viridis',
            s=100,
            alpha=0.7
        )

        plt.title('College Clusters in 2D PCA Space')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend(title='Cluster')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

print("✅ CollegeModel class defined successfully!")

class Recommender:
    def __init__(self, student_model, college_model):
        """
        Initialize the recommender with student and college models

        Parameters:
        -----------
        student_model : StudentModel
            Trained student model
        college_model : CollegeModel
            Trained college model
        """
        self.student_model = student_model
        self.college_model = college_model

    def recommend(self, student_dict, top_n=5):
        # Predict student cluster and get PCA features
        student_cluster, student_pca = self.student_model.predict_cluster(student_dict)

        # Get all colleges data with PCA features
        colleges_data = self.college_model.original_data
        college_pca_features = self.college_model.pca_features

        # Calculate similarity
        n_components = min(5, student_pca.shape[1], college_pca_features.shape[1])
        student_pca_truncated = student_pca[0, :n_components].reshape(1, -1)
        college_pca_truncated = college_pca_features[:, :n_components]
        similarities = cosine_similarity(student_pca_truncated, college_pca_truncated)[0]

        # Add similarity scores
        colleges_with_scores = colleges_data.copy()
        colleges_with_scores['Similarity'] = similarities

        # Filter by preferences
        filtered_colleges = self._filter_by_preferences(colleges_with_scores, student_dict)

        # Split into preferred location and others
        if 'Preferred Location' in student_dict and student_dict['Preferred Location']:
            location_pref = student_dict['Preferred Location'].strip().lower()
            preferred_mask = (
                (filtered_colleges['State'].str.lower() == location_pref) |
                (filtered_colleges['Location'].str.lower() == location_pref)
            )
            preferred_colleges = filtered_colleges[preferred_mask]
            other_colleges = filtered_colleges[~preferred_mask]
        else:
            preferred_colleges = pd.DataFrame()
            other_colleges = filtered_colleges

        # Get recommendations - top 3 from preferred, top 2 from others
        pref_rec = preferred_colleges.sort_values('Similarity', ascending=False).head(3)
        other_rec = other_colleges.sort_values('Similarity', ascending=False).head(2)

        # Combine recommendations
        recommendations = pd.concat([pref_rec, other_rec]).head(top_n)

        # If we don't have enough preferred colleges, fill with others
        if len(recommendations) < top_n:
            additional_needed = top_n - len(recommendations)
            additional = other_colleges.sort_values('Similarity', ascending=False).head(additional_needed)
            recommendations = pd.concat([recommendations, additional]).head(top_n)

        return recommendations

    def _filter_by_preferences(self, colleges_df, student_dict):
        filtered_df = colleges_df.copy()

        # Budget filtering with priority to closer matches
        if 'Budget' in student_dict and student_dict['Budget'] > 0:
            budget = student_dict['Budget']

            def get_fee_value(fee_str):
                if pd.isna(fee_str) or not isinstance(fee_str, str):
                    return np.nan
                try:
                    if 'LPA' in fee_str:
                        return float(fee_str.replace('LPA', '').replace('₹', '').strip()) * 100000
                    else:
                        return float(''.join(filter(str.isdigit, fee_str)))
                except:
                    return np.nan

            # Calculate fee values and differences from budget
            filtered_df['Fee_Value'] = filtered_df['Course Fees (₹)'].apply(get_fee_value)
            filtered_df['Budget_Diff'] = abs(filtered_df['Fee_Value'] - budget)

            # Filter out colleges way over budget (more than 20% over)
            filtered_df = filtered_df[
                (filtered_df['Fee_Value'] <= budget * 1.2) |
                (filtered_df['Fee_Value'].isna())
            ]

            # Sort by budget difference (closest to budget first)
            filtered_df = filtered_df.sort_values('Budget_Diff', ascending=True)

        # Location filtering
        if 'Preferred Location' in student_dict and student_dict['Preferred Location']:
            location_pref = student_dict['Preferred Location'].strip().lower()
            if location_pref:
                # Exact match for state or location
                location_mask = (
                    (filtered_df['State'].str.lower() == location_pref) |
                    (filtered_df['Location'].str.lower() == location_pref)
                )

                # If no exact matches, try partial matches
                if location_mask.sum() == 0:
                    location_mask = (
                        filtered_df['State'].str.lower().str.contains(location_pref, na=False) |
                        filtered_df['Location'].str.lower().str.contains(location_pref, na=False)
                    )

                filtered_df = filtered_df[location_mask]

        return filtered_df

print("✅ Recommender class defined successfully!")

# Create sample data function
def create_sample_data():
    """Create sample datasets if real files are not available"""

    print("📝 Creating sample student data...")
    student_data = pd.DataFrame({
        'Marks_10th': np.random.normal(80, 10, 100).clip(60, 100),
        'Marks_12th': np.random.normal(82, 10, 100).clip(60, 100),
        'JEE_Score': np.random.normal(120, 30, 100).clip(0, 300),
        'Budget': np.random.normal(500000, 200000, 100).clip(100000, 2000000),
        'Preferred Location': np.random.choice(['Karnataka', 'Delhi', 'Maharashtra', 'Tamil Nadu'], 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Target Exam': np.random.choice(['JEE', 'NEET', 'CUET'], 100),
        'State Board': np.random.choice(['CBSE', 'ICSE', 'State Board'], 100),
        'Category': np.random.choice(['General', 'OBC', 'SC', 'ST'], 100),
        'Stress Tolerance': np.random.choice(['Low', 'Average', 'High'], 100),
        'English Proficiency': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], 100),
        'Extra Curriculars': np.random.choice(['Sports, Music', 'Debate, Drama', 'Coding, Robotics', 'Art, Photography'], 100),
        'Future Goal': np.random.choice(['Engineering Career', 'Medical Career', 'Management Career', 'Research Career'], 100),
        'Certifications': np.random.choice(['Programming', 'Data Science', 'Web Development', 'Digital Marketing'], 100)
    })

    print("📝 Creating sample college data...")
    college_data = pd.DataFrame({
        'College Name': [
        'Delhi Technological University', 'Indian Institute of Technology Delhi',
        'Netaji Subhas University of Technology', 'Jamia Millia Islamia',
        'University of Delhi', 'Indian Institute of Technology Bombay',
        'Veermata Jijabai Technological Institute', 'College of Engineering Pune',
        # Add more real college names
    ],
        'Location': np.random.choice(['Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Pune', 'Hyderabad'], 50),
        'State': np.random.choice(['Karnataka', 'Delhi', 'Maharashtra', 'Tamil Nadu', 'Telangana'], 50),
        'College Type': np.random.choice(['Government', 'Private', 'Autonomous'], 50),
        'Established Year': np.random.randint(1950, 2020, 50),
        'Approved By': np.random.choice(['AICTE', 'UGC', 'NBA'], 50),
        'NIRF Ranking': np.random.randint(1, 200, 50),
        'Course Fees (₹)': [f'{np.random.uniform(2, 10):.1f} LPA' for _ in range(50)],
        'Notable Courses Offered': np.random.choice([
            'Engineering, Management', 'Engineering, Science', 'Management, Commerce',
            'Engineering, Technology', 'Science, Research'
        ], 50),
        'Education Loan': np.random.choice(['Yes', 'No'], 50),
        'Placement (Average': [f'{np.random.uniform(5, 15):.1f} LPA' for _ in range(50)]
    })

    return student_data, college_data

# Main execution
def main():
    print("=== LOADING DATASETS ===")

    # Try to load real datasets, fallback to sample data
    try:
        student_data = pd.read_excel("/content/student_dataset00.xlsx")
        print(f"✅ Student dataset loaded: {student_data.shape[0]} records, {student_data.shape[1]} features")
    except Exception as e:
        print(f"⚠️ Could not load student_dataset00.xlsx: {e}")
        student_data, _ = create_sample_data()
        print(f"✅ Created sample student data: {student_data.shape}")

    try:
        college_data = pd.read_excel("/content/210colleges_dataset_krip.ai.xlsx")
        print(f"✅ College dataset loaded: {college_data.shape[0]} records, {college_data.shape[1]} features")
    except Exception as e:
        print(f"⚠️ Could not load 210colleges_dataset_krip.ai.xlsx: {e}")
        _, college_data = create_sample_data()
        print(f"✅ Created sample college data: {college_data.shape}")

    print("\n=== TRAINING MODELS ===")

    # Train Student Model
    print("\n1. Training Student Model...")
    start_time = datetime.now()
    student_model = StudentModel(num_clusters=5, random_state=42)
    student_model.fit(student_data)
    training_time = datetime.now() - start_time
    print(f"✅ Student model trained in {training_time.total_seconds():.2f} seconds")

    # Train College Model
    print("\n2. Training College Model...")
    start_time = datetime.now()
    college_model = CollegeModel(num_clusters=8, random_state=42)
    college_model.fit(college_data)
    training_time = datetime.now() - start_time
    print(f"✅ College model trained in {training_time.total_seconds():.2f} seconds")

    # Create Recommender
    print("\n3. Creating Recommender System...")
    recommender = Recommender(student_model, college_model)
    print("✅ Recommender system created successfully!")

    # Test the system
    print("\n=== TESTING RECOMMENDATION SYSTEM ===")
    test_student = {
        'Marks_10th': 85,
        'Marks_12th': 82,
        'JEE_Score': 125,
        'Budget': 350000,
        'Preferred Location': 'Karnataka',
        'Gender': 'Male',
        'Certifications': 'Python, AI',
        'Target Exam': 'JEE',
        'State Board': 'CBSE',
        'Category': 'General',
        'Stress Tolerance': 'High',
        'English Proficiency': 'Excellent',
        'Extra Curriculars': 'Coding, Sports',
        'Future Goal': 'AI Engineer'
    }

    try:
        recommendations = recommender.recommend(test_student, top_n=5)
        print(f"✅ Generated {len(recommendations)} recommendations")

        print(f"\nTop Recommendations:")
        for i, (_, row) in enumerate(recommendations.iterrows()):
            print(f"{i+1}. {row['College Name']} - Similarity: {row['Similarity']:.3f}")

    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")

    # Save models
    print("\n=== SAVING MODELS ===")
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    try:
        # Save models
        with open(os.path.join(models_dir, "student_model.pkl"), 'wb') as f:
            pickle.dump(student_model, f)
        print("✅ Student model saved")

        with open(os.path.join(models_dir, "college_model.pkl"), 'wb') as f:
            pickle.dump(college_model, f)
        print("✅ College model saved")

        with open(os.path.join(models_dir, "recommender.pkl"), 'wb') as f:
            pickle.dump(recommender, f)
        print("✅ Recommender saved")

        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'student_clusters': student_model.num_clusters,
            'college_clusters': college_model.num_clusters,
            'num_students': len(student_data),
            'num_colleges': len(college_data),
            'text_embedding_method': 'SentenceTransformer' if student_model.text_embedder.use_sentence_transformers else 'TF-IDF'
        }

        with open(os.path.join(models_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        print("✅ Metadata saved")

    except Exception as e:
        print(f"❌ Error saving models: {e}")

    print(f"\n🎉 Setup completed successfully!")
    print(f"📁 Models saved in '{models_dir}' directory")

    return student_model, college_model, recommender

# Simple function for Gradio interface
def get_recommendations_simple(marks_10th, marks_12th, jee_score, budget,
                              preferred_location, gender, certifications, target_exam,
                              state_board, category, stress_tolerance, english_proficiency,
                              extra_curriculars, future_goal, top_n=5):
    """
    Simple function to get recommendations - works with saved models
    """
    try:
        # Load models (in real app, do this once at startup)
        with open('models/recommender.pkl', 'rb') as f:
            recommender = pickle.load(f)

        student_dict = {
            'Marks_10th': marks_10th,
            'Marks_12th': marks_12th,
            'JEE_Score': jee_score,
            'Budget': budget,
            'Preferred Location': preferred_location,
            'Gender': gender,
            'Certifications': certifications,
            'Target Exam': target_exam,
            'State Board': state_board,
            'Category': category,
            'Stress Tolerance': stress_tolerance,
            'English Proficiency': english_proficiency,
            'Extra Curriculars': extra_curriculars,
            'Future Goal': future_goal
        }

        recommendations = recommender.recommend(student_dict, top_n=top_n)

        # Format results
        results = []
        for i, (_, row) in enumerate(recommendations.iterrows()):
            results.append({
                'Rank': i + 1,
                'College': row['College Name'],
                'Location': f"{row['Location']}, {row['State']}",
                'Type': row['College Type'],
                'Fees': row['Course Fees (₹)'],
                'Similarity': f"{row['Similarity']:.3f}"
            })

        return results

    except Exception as e:
        return [{'Error': str(e)}]

# Create Gradio app code
gradio_app_code = '''
import gradio as gr
import pickle
import pandas as pd

# Load models once at startup
print("Loading models...")
try:
    with open('models/student_model.pkl', 'rb') as f:
        student_model = pickle.load(f)

    with open('models/college_model.pkl', 'rb') as f:
        college_model = pickle.load(f)

    with open('models/recommender.pkl', 'rb') as f:
        recommender = pickle.load(f)

    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Make sure you have run the training script first!")

def get_recommendations(marks_10th, marks_12th, jee_score, budget,
                       preferred_location, gender, certifications, target_exam,
                       state_board, category, stress_tolerance, english_proficiency,
                       extra_curriculars, future_goal):
    """Get college recommendations for a student"""

    student_dict = {
        'Marks_10th': marks_10th,
        'Marks_12th': marks_12th,
        'JEE_Score': jee_score,
        'Budget': budget,
        'Preferred Location': preferred_location,
        'Gender': gender,
        'Certifications': certifications,
        'Target Exam': target_exam,
        'State Board': state_board,
        'Category': category,
        'Stress Tolerance': stress_tolerance,
        'English Proficiency': english_proficiency,
        'Extra Curriculars': extra_curriculars,
        'Future Goal': future_goal
    }

    try:
        recommendations = recommender.recommend(student_dict, top_n=5)

        # Format as HTML table
        html = "<div style='font-family: Arial, sans-serif;'>"
        html += "<h2 style='color: #2E86AB; text-align: center;'>🎓 Your Top 5 College Recommendations</h2>"

        if len(recommendations) == 0:
            html += "<p style='text-align: center; color: #E74C3C;'>No recommendations found. Please try adjusting your criteria.</p>"
        else:
            html += "<table style='width: 100%; border-collapse: collapse; margin: 20px 0;'>"
            html += "<thead><tr style='background-color: #34495E; color: white;'>"
            html += "<th style='padding: 12px; border: 1px solid #ddd;'>Rank</th>"
            html += "<th style='padding: 12px; border: 1px solid #ddd;'>College Name</th>"
            html += "<th style='padding: 12px; border: 1px solid #ddd;'>Location</th>"
            html += "<th style='padding: 12px; border: 1px solid #ddd;'>Type</th>"
            html += "<th style='padding: 12px; border: 1px solid #ddd;'>Fees</th>"
            html += "<th style='padding: 12px; border: 1px solid #ddd;'>Match Score</th>"
            html += "</tr></thead><tbody>"

            for i, (_, row) in enumerate(recommendations.iterrows()):
                bg_color = "#ECF0F1" if i % 2 == 0 else "#FFFFFF"
                html += f"<tr style='background-color: {bg_color};'>"
                html += f"<td style='padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: bold;'>{i+1}</td>"
                html += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['College Name']}</td>"
                html += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Location']}, {row['State']}</td>"
                html += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['College Type']}</td>"
                html += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Course Fees (₹)']}</td>"
                html += f"<td style='padding: 10px; border: 1px solid #ddd; text-align: center;'><span style='background-color: #2ECC71; color: white; padding: 4px 8px; border-radius: 4px;'>{row['Similarity']:.3f}</span></td>"
                html += "</tr>"

            html += "</tbody></table>"

            # Add explanation
            html += "<div style='background-color: #EBF3FD; padding: 15px; border-radius: 8px; margin-top: 20px;'>"
            html += "<h4 style='color: #2E86AB; margin-top: 0;'>📊 How Recommendations Work:</h4>"
            html += "<ul style='color: #34495E;'>"
            html += "<li><strong>Match Score:</strong> Higher scores indicate better compatibility with your profile</li>"
            html += "<li><strong>Budget Filter:</strong> Colleges within your specified budget range</li>"
            html += "<li><strong>Location Preference:</strong> Priority given to your preferred location</li>"
            html += "<li><strong>Academic Fit:</strong> Based on your marks and exam scores</li>"
            html += "</ul></div>"

        html += "</div>"
        return html

    except Exception as e:
        error_html = f"<div style='color: #E74C3C; padding: 20px; text-align: center;'>"
        error_html += f"<h3>❌ Error generating recommendations</h3>"
        error_html += f"<p>{str(e)}</p>"
        error_html += f"<p><em>Please make sure all models are properly loaded.</em></p>"
        error_html += f"</div>"
        return error_html

# Create the Gradio interface
with gr.Blocks(
    title="College Recommendation System",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    """
) as app:

    gr.Markdown("""
    # 🎓 AI-Powered College Recommendation System

    Get personalized college recommendations based on your academic performance, preferences, and career goals!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Academic Information")
            marks_10th = gr.Slider(
                minimum=60, maximum=100, value=85, step=0.1,
                label="10th Standard Marks (%)",
                info="Your 10th standard percentage"
            )
            marks_12th = gr.Slider(
                minimum=60, maximum=100, value=80, step=0.1,
                label="12th Standard Marks (%)",
                info="Your 12th standard percentage"
            )
            jee_score = gr.Slider(
                minimum=0, maximum=300, value=120, step=1,
                label="Entrance Exam Score (JEE/NEET/CUET)",
                info="Your entrance exam score"
            )
            budget = gr.Number(
                value=500000, minimum=50000, maximum=5000000,
                label="Budget (₹ per year)",
                info="Maximum fee you can afford per year"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Preferences & Background")
            preferred_location = gr.Textbox(
                value="Karnataka",
                label="Preferred Location",
                info="State or city preference (e.g., Karnataka, Delhi)"
            )
            gender = gr.Dropdown(
                choices=["Male", "Female", "Other"],
                value="Male",
                label="Gender"
            )
            target_exam = gr.Dropdown(
                choices=["JEE", "NEET", "CUET", "CAT", "Other"],
                value="JEE",
                label="Target Entrance Exam"
            )
            state_board = gr.Dropdown(
                choices=["CBSE", "ICSE", "State Board", "IB", "Other"],
                value="CBSE",
                label="Education Board"
            )

    with gr.Row():
        category = gr.Dropdown(
            choices=["General", "OBC", "SC", "ST", "EWS"],
            value="General",
            label="Category"
        )
        stress_tolerance = gr.Dropdown(
            choices=["Low", "Average", "High"],
            value="Average",
            label="Stress Tolerance Level"
        )
        english_proficiency = gr.Dropdown(
            choices=["Poor", "Average", "Good", "Excellent"],
            value="Good",
            label="English Proficiency"
        )

    with gr.Row():
        with gr.Column():
            certifications = gr.Textbox(
                value="Programming, Web Development",
                label="Certifications & Skills",
                info="List your certifications and technical skills",
                lines=2
            )
        with gr.Column():
            extra_curriculars = gr.Textbox(
                value="Sports, Music, Debate",
                label="Extra Curricular Activities",
                info="Your hobbies and activities outside academics",
                lines=2
            )

    future_goal = gr.Textbox(
        value="Software Engineer at a top tech company",
        label="Career Goals & Aspirations",
        info="Describe your future career plans and goals",
        lines=3
    )

    with gr.Row():
        submit_btn = gr.Button(
            "🔍 Get My College Recommendations",
            variant="primary",
            size="lg"
        )
        clear_btn = gr.Button(
            "🔄 Clear All",
            variant="secondary"
        )

    output = gr.HTML(label="Recommendations")

    # Event handlers
    submit_btn.click(
        fn=get_recommendations,
        inputs=[
            marks_10th, marks_12th, jee_score, budget, preferred_location,
            gender, certifications, target_exam, state_board, category,
            stress_tolerance, english_proficiency, extra_curriculars, future_goal
        ],
        outputs=output
    )

    def clear_all():
        return (
            85, 80, 120, 500000, "Karnataka", "Male", "Programming, Web Development",
            "JEE", "CBSE", "General", "Average", "Good", "Sports, Music, Debate",
            "Software Engineer at a top tech company", ""
        )

    clear_btn.click(
        fn=clear_all,
        outputs=[
            marks_10th, marks_12th, jee_score, budget, preferred_location,
            gender, certifications, target_exam, state_board, category,
            stress_tolerance, english_proficiency, extra_curriculars, future_goal, output
        ]
    )

    gr.Markdown("""
    ---
    ### 💡 Tips for Better Recommendations:
    - **Be Accurate**: Provide accurate academic scores for better matching
    - **Be Specific**: Detailed career goals help find relevant programs
    - **Budget Realistic**: Set a realistic budget for more practical options
    - **Location Flexible**: Consider multiple locations for more choices

    ### 🔍 About the System:
    This AI system uses machine learning to analyze your profile and match it with colleges based on:
    - Academic performance and entrance scores
    - Financial constraints and budget
    - Location preferences and accessibility
    - Career goals and program alignment
    - Personal interests and extracurricular activities
    """)

if __name__ == "__main__":
    print("🚀 Starting College Recommendation System...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
'''

def save_gradio_app():
    """Save the Gradio app code to a file"""
    try:
        with open("gradio_app.py", "w", encoding='utf-8') as f:
            f.write(gradio_app_code)
        print("✅ Gradio app saved to 'gradio_app.py'")
        return True
    except Exception as e:
        print(f"❌ Error saving Gradio app: {e}")
        return False

if __name__ == "__main__":
    # Run the main training and setup
    student_model, college_model, recommender = main()

    # Save the Gradio app
    save_gradio_app()

    print(f"\n{'='*60}")
    print("🎉 COMPLETE SETUP FINISHED!")
    print('='*60)
    print("📁 Files created:")
    print("   ├── models/student_model.pkl")
    print("   ├── models/college_model.pkl")
    print("   ├── models/recommender.pkl")
    print("   ├── models/metadata.pkl")
    print("   └── gradio_app.py")
    print()
    print("🚀 To run the web interface:")
    print("   python gradio_app.py")
    print()
    print("✅ The system now works offline with TF-IDF fallback!")
    print("   No internet connection required for sentence transformers")
