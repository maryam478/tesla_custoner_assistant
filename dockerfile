# Use official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src

# Expose the Streamlit port
EXPOSE 8501

# Set Streamlit to run your UI file
CMD ["streamlit", "run", "src/tesla_ui.py"]
