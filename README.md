# NLP Copy Detector

A simple Natural Language Processing-based application that detects potential plagiarism between multiple text documents using semantic similarity.

## Features

* Computes document similarity using NLP techniques
* Generates a similarity heatmap for easy visualization
* Includes pre-trained model for efficient comparison
* Supports input from plain text files (`.txt`)
* Easy to use via command line

## Project Structure

```
NLP-Copy-Detector-main/
│
├── Copy_Detector.py           # Main script to run detection
├── nltk_setup.py              # Downloads required NLTK resources
├── model.bin                  # Pre-trained vector model (e.g., Word2Vec or Doc2Vec)
├── requirements.txt           # Python dependencies
├── similarity_heatmap.png     # Sample output heatmap image
├── assignments/               # Folder with sample assignment text files
├── LICENSE                    # Project license file
└── README.md                  # Project documentation
```

## Requirements

* Python 3.7+
* `gensim`, `nltk`, `matplotlib`, `scikit-learn`, `seaborn`

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## NLTK Resources

Before first run, download required NLTK data:

```bash
python nltk_setup.py
```

## How to Use

1. Add your text files in the `assignments/` folder.
2. Run the main script:

```bash
python Copy_Detector.py
```

3. View the similarity matrix and generated heatmap (`similarity_heatmap.png`).

## Example Output

The tool outputs a similarity matrix comparing every document pair and visualizes it in a heatmap format for quick inspection.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
