from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def calculate_tfidf(corpus):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Extract feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert TF-IDF matrix to dense format for easier viewing
    dense_matrix = tfidf_matrix.todense()
    
    # Display TF-IDF values for each document
    for i, doc in enumerate(dense_matrix):
        print(f"\nDocument {i + 1} TF-IDF Scores:")
        scores = np.array(doc).flatten()
        for word, score in zip(feature_names, scores):
            if score > 0:
                print(f"  {word}: {score:.4f}")

# Main function to handle user input and execute TF-IDF calculation
def main():
    # Accept user input for the number of documents
    n = int(input("Enter the number of documents: "))
    corpus = [input(f"Enter document {i+1}: ") for i in range(n)]
    
    # Calculate and display TF-IDF scores
    calculate_tfidf(corpus)

# Run the program
if __name__ == "_main_":
    main()




