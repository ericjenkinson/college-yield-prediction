# Deploying to Google Cloud Platform

This guide shows you how to host your Docker image on Google Artifact Registry (GAR) and deploy it to Cloud Run.

## Prerequisites
- `gcloud` CLI installed and authenticated.
- Docker running locally.
- A Google Cloud Project.

## Steps

### 1. Set Environment Variables
Open your terminal and set these variables for convenience. Replace `[YOUR_PROJECT_ID]` with your actual project ID (e.g., from `gcloud projects list`).

```bash
export PROJECT_ID="college-yield-predictor"
export REGION="us-central1"
export REPO_NAME="yield-predictor-repo"
export IMAGE_NAME="yield-predictor"
export TAG="latest"
```

### 2. Configure gcloud
Ensure you are working with the correct project:
```bash
gcloud config set project $PROJECT_ID
```

### 3. Enable Artifact Registry API
Enable the necessary API to store container images:
```bash
gcloud services enable artifactregistry.googleapis.com
```

### 4. Create a Docker Repository
Create a repository in Artifact Registry to store your images:
```bash
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Yield Predictor"
```

### 5. Configure Docker Authentication
Configure Docker to authenticate with your Google Cloud region:
```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

### 6. Build and Tag Your Image
Since Cloud Run requires `linux/amd64`, if you are on an Apple Silicon Mac, you must build with the platform flag:
```bash
docker build --platform linux/amd64 -t yield-predictor .
```

Then tag it:
```bash
docker tag yield-predictor:latest ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}
```

### 7. Push the Image
Push the tagged image to Artifact Registry:
```bash
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}
```

### 8. (Optional) Deploy to Cloud Run
To run your image as a web service (Serverless):
```bash
gcloud run deploy yield-service \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG} \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 50001 \
    --memory 2Gi
```
*Note: We verify the port is 50001 based on your recent changes.*

## Verification
1. Go to the [Google Cloud Console Artifact Registry](https://console.cloud.google.com/artifacts) to see your image.
2. If deployed to Cloud Run, visit the provided URL to test the API.
