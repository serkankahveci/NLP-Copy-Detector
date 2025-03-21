import os
import nltk
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources with error handling
def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.error(f"Failed to download NLTK resource {resource}: {str(e)}")
            raise

def preprocess_text(text):
    """Advanced text preprocessing with better cleaning and normalization"""
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    
    # Tokenize, remove stopwords and lemmatize
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
    
    return ' '.join(words)

def load_documents_from_files(file_paths):
    """Load documents with better error handling and progress tracking"""
    documents = []
    file_names = []
    
    logger.info(f"Processing {len(file_paths)} documents...")
    
    for file_path in tqdm(file_paths, desc="Loading files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                document_text = file.read()
                
                # Skip empty documents
                if not document_text.strip():
                    logger.warning(f"Empty document: {file_path}")
                    continue
                    
                cleaned_text = preprocess_text(document_text)
                
                # Skip documents with too few words after preprocessing
                if len(cleaned_text.split()) < 20:
                    logger.warning(f"Document too short after preprocessing: {file_path}")
                    continue
                    
                documents.append(cleaned_text)
                file_names.append(os.path.basename(file_path))
                
        except UnicodeDecodeError:
            logger.error(f"Unicode decode error in {file_path}. Trying with different encoding...")
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    document_text = file.read()
                    cleaned_text = preprocess_text(document_text)
                    documents.append(cleaned_text)
                    file_names.append(os.path.basename(file_path))
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {str(e)}")
                
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents, file_names

def train_doc2vec_model(documents, file_names, save_path="model.bin"):
    """Train a new Doc2Vec model if needed"""
    tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(documents)]
    
    logger.info("Training Doc2Vec model...")
    model = Doc2Vec(vector_size=100, 
                    min_count=2,
                    dm=1,
                    epochs=40,
                    window=10,
                    workers=4)
    
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    logger.info(f"Saving model to {save_path}")
    model.save(save_path)
    
    return model

def load_or_train_model(documents, file_names, model_path="model.bin"):
    """Load existing model or train a new one if not available"""
    try:
        logger.info(f"Attempting to load model from {model_path}")
        model = Doc2Vec.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except:
        logger.info("No existing model found. Training new model...")
        return train_doc2vec_model(documents, file_names, model_path)

def compute_document_vectors(model, documents):
    """Compute document vectors with chunking for long documents"""
    document_vectors = []
    
    logger.info("Computing document vectors...")
    
    for doc in tqdm(documents, desc="Vectorizing"):
        words = doc.split()
        
        # For very long documents, chunk and average vectors
        if len(words) > 5000:
            chunks = [words[i:i+5000] for i in range(0, len(words), 5000)]
            vectors = [model.infer_vector(chunk) for chunk in chunks]
            avg_vector = np.mean(vectors, axis=0)
            document_vectors.append(avg_vector)
        else:
            document_vectors.append(model.infer_vector(words))
            
    return np.array(document_vectors)

def analyze_similarities(similarities, file_names, threshold=0.75):
    """Analyze similarities with multiple thresholds"""
    results = defaultdict(list)
    
    # Check different thresholds
    thresholds = [0.65, 0.75, 0.85, 0.95]
    
    for t in thresholds:
        count = 0
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i][j] > t:
                    results[t].append((file_names[i], file_names[j], similarities[i][j]))
                    count += 1
        logger.info(f"Found {count} document pairs with similarity > {t}")
    
    return results

def generate_similarity_matrix(similarity_matrix, file_names, output_path="similarity_heatmap.png"):
    """Generate and save similarity heatmap"""
    plt.figure(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
    
    # Create a custom colormap that emphasizes high similarity
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Plot the heatmap
    ax = sns.heatmap(
        similarity_matrix,
        mask=mask,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        xticklabels=[os.path.splitext(name)[0] for name in file_names],
        yticklabels=[os.path.splitext(name)[0] for name in file_names]
    )
    
    plt.title("Document Similarity Matrix", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    logger.info(f"Saving similarity heatmap to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return output_path

def generate_detailed_report(similarity_results, file_names, documents, output_path="plagiarism_report.html"):
    """Generate detailed HTML report with similarity analysis"""
    import datetime
    
    # Create report dataframe
    rows = []
    for threshold, pairs in similarity_results.items():
        for doc1, doc2, score in pairs:
            rows.append({
                "Document 1": doc1,
                "Document 2": doc2,
                "Similarity Score": score,
                "Threshold": threshold,
                "Status": "Potential Plagiarism" if score > 0.85 else "Suspicious"
            })
    
    if not rows:
        logger.info("No suspicious document pairs found")
        return None
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Similarity Score", ascending=False)
    
    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plagiarism Detection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .high-similarity {{ background-color: #ffcccc; }}
            .warning {{ color: #e74c3c; font-weight: bold; }}
            .info {{ color: #3498db; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Plagiarism Detection Report</h1>
        <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Documents analyzed: {len(file_names)}</p>
        
        <h2>Similarity Analysis Results</h2>
        <table>
            <tr>
                <th>Document 1</th>
                <th>Document 2</th>
                <th>Similarity Score</th>
                <th>Status</th>
            </tr>
    """
    
    for _, row in df.iterrows():
        html += f"""
            <tr class="{'high-similarity' if row['Similarity Score'] > 0.85 else ''}">
                <td>{row['Document 1']}</td>
                <td>{row['Document 2']}</td>
                <td>{row['Similarity Score']:.4f}</td>
                <td>{row['Status']}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Interpretation</h2>
        <ul>
            <li><span class="warning">Potential Plagiarism (> 0.85):</span> High similarity suggesting potential plagiarism. Requires manual review.</li>
            <li><span class="info">Suspicious (0.65 - 0.85):</span> Moderate similarity that may indicate shared concepts or partial copying.</li>
        </ul>
        
        <h2>Similarity Heatmap</h2>
        <img src="similarity_heatmap.png" alt="Similarity Heatmap">
        
        <p><small>Note: This report is generated automatically and similarity scores should be verified through manual inspection.</small></p>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    logger.info(f"Detailed report saved to {output_path}")
    return output_path

def extract_common_passages(doc1, doc2, window_size=10):
    """Extract potentially copied passages between documents"""
    sentences1 = sent_tokenize(doc1)
    sentences2 = sent_tokenize(doc2)
    
    matches = []
    
    # Use sliding window of words for comparison
    for i in range(len(sentences1)):
        words1 = word_tokenize(sentences1[i].lower())
        
        if len(words1) < window_size:
            continue
            
        for j in range(len(sentences2)):
            words2 = word_tokenize(sentences2[j].lower())
            
            if len(words2) < window_size:
                continue
                
            # Check for matching word sequences
            for k in range(len(words1) - window_size + 1):
                window1 = words1[k:k+window_size]
                
                for l in range(len(words2) - window_size + 1):
                    window2 = words2[l:l+window_size]
                    
                    if window1 == window2:
                        matches.append({
                            'passage': ' '.join(window1),
                            'doc1_context': sentences1[i],
                            'doc2_context': sentences2[j]
                        })
                        break
    
    # Remove duplicates
    unique_matches = []
    seen_passages = set()
    
    for match in matches:
        if match['passage'] not in seen_passages:
            seen_passages.add(match['passage'])
            unique_matches.append(match)
    
    return unique_matches

def main():
    """Main function to run the plagiarism detection system"""
    try:
        # Download NLTK resources
        download_nltk_resources()
        
        # Get file paths from the 'assignments' directory
        if not os.path.exists("assignments"):
            logger.error("Assignments directory not found. Creating it...")
            os.makedirs("assignments")
            logger.info("Please add assignment files to the 'assignments' directory and run again.")
            return
            
        assignments = [os.path.join("assignments", file) for file in os.listdir("assignments")
                    if os.path.isfile(os.path.join("assignments", file)) and file.endswith((".txt", ".doc", ".docx", ".pdf"))]
        
        if not assignments:
            logger.error("No assignment files found in the 'assignments' directory.")
            return
            
        # Load documents
        documents, file_names = load_documents_from_files(assignments)
        
        if len(documents) < 2:
            logger.error("At least 2 valid documents are needed for comparison.")
            return
            
        # Load or train model
        model = load_or_train_model(documents, file_names)
        
        # Compute document vectors
        document_vectors = compute_document_vectors(model, documents)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(document_vectors)
        
        # Print similarity between documents
        logger.info("Document similarity scores:")
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i][j]
                logger.info(f"Similarity between {file_names[i]} and {file_names[j]}: {similarity:.4f}")
        
        # Analyze similarities at different thresholds
        similarity_results = analyze_similarities(similarity_matrix, file_names)
        
        # Generate similarity matrix visualization
        generate_similarity_matrix(similarity_matrix, file_names)
        
        # Generate detailed report
        report_path = generate_detailed_report(similarity_results, file_names, documents)
        
        if report_path:
            logger.info("\nPlagiarism detection complete!")
            logger.info(f"Results saved to {report_path}")
            
            # Print suspicious documents
            high_similarity_pairs = similarity_results[0.85]
            if high_similarity_pairs:
                logger.warning("\nPotential plagiarism detected:")
                for doc1, doc2, score in high_similarity_pairs:
                    logger.warning(f"Document {doc1} and Document {doc2}: {score:.4f}")
        else:
            logger.info("No suspicious similarity detected between documents.")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()