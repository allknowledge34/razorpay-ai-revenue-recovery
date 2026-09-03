FROM python:3.13-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary project directories and files
COPY app/ app/
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY docs/ docs/
COPY reports/ reports/
COPY README.md .

# Configure Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    PYTHONPATH=/app

# Expose port
EXPOSE 8501

# Healthcheck using Python's built-in urllib to avoid installing curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Start the application
CMD ["streamlit", "run", "app/streamlit_app.py"]
