FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better cache usage
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["streamlit", "run", "Data_Analysis_App📊.py", "--server.address", "0.0.0.0"]
