FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3 -m pip install --upgrade pip

# Copy project
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit listen on port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
ENTRYPOINT ["bash", "-c", "python create_model.py && streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
