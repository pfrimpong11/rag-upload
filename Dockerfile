FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (no git needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY nltk_data /app/nltk_data

# Create cache and tmp directories with appropriate permissions
RUN mkdir -p /app/cache /app/tmp \
    && chmod -R 755 /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]