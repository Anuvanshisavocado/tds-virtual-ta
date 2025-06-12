# Stage 1: Build Environment - Use a builder image to compile dependencies
# This keeps the final image small by not including build tools.
FROM python:3.9-slim-buster AS builder

# Set environment variables to prevent interactive prompts during apt-get installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Tesseract OCR and its English language data
# 'tesseract-ocr' is the main binary
# 'tesseract-ocr-eng' is the English language pack. Adjust if you need other languages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        # Add any other core system dependencies Tesseract might implicitly need, e=g., libgomp1 for some FAISS builds.
        # Although --no-install-recommends should help, sometimes hidden dependencies appear.
        # For a truly minimal setup, you might need to test and add specific lib packages.
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the Tesseract data directory explicitly if needed (often not necessary if installed globally)
# ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements.txt first to leverage Docker's build cache
COPY requirements.txt .

# Install Python dependencies using pip, no caching to save space
# Using --default-timeout to prevent issues on slow connections
# Consider pre-installing numpy/scipy if they are known to be problematic, then the rest.
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Stage 2: Runtime Environment - Create a super-slim final image
# Use the same slim base image for consistency and smallest size
FROM python:3.9-slim-buster

# Copy the Tesseract OCR binaries and language data from the builder stage
# This ensures only the *installed* Tesseract components are present, not the build tools.
COPY --from=builder /usr/bin/tesseract /usr/bin/tesseract
COPY --from=builder /usr/share/tesseract-ocr /usr/share/tesseract-ocr
COPY --from=builder /var/lib/tesseract-ocr /var/lib/tesseract-ocr # Some configs/data might be here

# Set the Tesseract data directory explicitly in the final image
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata # Confirm actual path after apt-get install

# Set environment variables for the application (from Render's UI later)
# ENV OPENAI_API_KEY="" # Set these via Render Dashboard
# ENV TESSERACT_CMD="/usr/bin/tesseract" # Set this via Render Dashboard

# Set the working directory for the application
WORKDIR /app

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy the rest of your application code
# .dockerignore should exclude unnecessary files (like .git, .env, __pycache__)
COPY . .

# Expose the port your FastAPI app runs on
EXPOSE 8000

# Command to run your application when the container starts
# Render's Start Command will override this, but it's good practice to include.
# Ensure data_processor.py executes successfully before uvicorn starts.
CMD ["bash", "-c", "python data_processor.py && uvicorn app:app --host 0.0.0.0 --port 8000"]
