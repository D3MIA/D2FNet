# Git LFS Setup Instructions for Force-Estimation-Models

## Step 1: Install Git LFS (if not already installed)
git lfs install

## Step 2: Add the .gitattributes file (already created)
git add .gitattributes
git commit -m "Configure Git LFS for model files"

## Step 3: Add your model files (they will be tracked by LFS automatically)
git add CNN_approach/models/
git add Gradient-Boosting_LightGBM_approach/models/
git commit -m "Add trained models (CNN and LightGBM) via Git LFS"

## Step 4: Push to GitHub
git push origin main

## Verification Commands:

# Check which files are tracked by LFS
git lfs ls-files

# Check LFS status
git lfs status

# See total LFS storage used
git lfs env

## Notes:
# - Total size: ~850 MB (within 1GB free tier)
# - Models remain private with your repository
# - .gitattributes ensures all *.pth and *.joblib files use LFS
# - Future model updates will automatically use LFS
