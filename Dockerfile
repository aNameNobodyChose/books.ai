FROM pytorch/pytorch:latest

# Install Python packages
RUN pip install --no-cache-dir nltk scikit-learn matplotlib

# Download NLTK resources (including 'punkt')
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Set the working directory
WORKDIR /workspace

# Default command
CMD ["python"]
