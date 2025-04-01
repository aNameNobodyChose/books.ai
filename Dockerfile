FROM pytorch/pytorch:latest

# Install NLTK and other dependencies
RUN pip install --no-cache-dir nltk scikit-learn

# Download NLTK resources (including 'punkt')
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Set the working directory
WORKDIR /workspace

# Default command
CMD ["python"]
