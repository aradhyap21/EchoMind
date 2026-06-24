FROM python:3.10-slim

# Install system dependencies (ffmpeg is required for librosa audio decoding, libgl1-mesa-glx for OpenCV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000 (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . .

# Expose port 7860 (Hugging Face Spaces standard port)
EXPOSE 7860

# Run the FastAPI server via Uvicorn, preloading models to avoid cold starts in production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
