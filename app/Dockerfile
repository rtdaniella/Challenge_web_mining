# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install the required dependencies
RUN apt-get update && \
    apt-get install -y build-essential python3-dev libatlas-base-dev
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download fr_core_news_sm
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy the current directory contents into the container at /app
COPY data /app/data
COPY . /app
# Expose the port that Streamlit will run on
EXPOSE 8501

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
