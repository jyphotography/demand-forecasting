# Use Python 3.12 slim version
FROM python:3.12-slim

# Install pipenv
RUN pip install pipenv

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies
RUN pipenv install --deploy --system

# Create models directory
RUN mkdir -p models

# Copy necessary files
COPY ["src/predict_api.py", "./"]
COPY ["models/random_forest_model.bin", "models/dict_vectorizer.bin", "./models/"]


# Expose port
EXPOSE 9696

# Run the API with gunicorn
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict_api:app"] 