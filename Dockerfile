FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install Python packages with specific versions
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    fastapi==0.110.0 \
    uvicorn==0.27.1 \
    nltk==3.8.1 \
    scikit-learn==1.3.2 \
    matplotlib==3.8.2 \
    spacy==3.5.0 \
    coreferee==1.4.1 \
    transformers==4.39.3 \
    openai==1.73.0 \
    datasets==2.18.0 \
    kneed==0.8.5

# Download NLTK resources
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy English language model
RUN python -m spacy download en_core_web_sm

# Install coreferee English model
RUN python -m coreferee install en

# Set the working directory
WORKDIR /workspace

# Default command
CMD ["python"]
