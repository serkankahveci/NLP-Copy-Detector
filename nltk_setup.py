import nltk

resources = [
    'punkt',
    'punkt_tab',
    'perluniprops',
    'nonbreaking_prefixes',
    'wordnet',
    'stopwords',
    'omw-1.4'
]

for res in resources:
    try:
        nltk.data.find(res)
        print(f"{res} already available.")
    except LookupError:
        print(f"Downloading {res}...")
        nltk.download(res)
