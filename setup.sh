#!/bin/bash
# setup.sh — Setup script for Qwen Image Editor

set -e

echo "🎨 Setting up Qwen Image Editor..."
echo ""

# Create directories
mkdir -p backend/uploads backend/outputs

echo "✅ Directories created"

# Check Python
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 found: $(python3 --version)"
else
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
cd backend
pip install -r requirements.txt
cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Setup complete!"
echo ""
echo "Before running, make sure:"
echo "  1. ComfyUI is running at http://127.0.0.1:8188"
echo "  2. Required models are installed:"
echo "     • qwen_image_vae.safetensors (VAE)"
echo "     • qwen_2.5_vl_7b_fp8_scaled.safetensors (CLIP)"
echo "     • qwen_image_edit_2509_fp8_e4m3fn.safetensors (UNET)"
echo "     • Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors (LoRA)"
echo ""
echo "To start:"
echo "  cd backend && uvicorn main:app --reload --port 8000"
echo ""
echo "Then open frontend/index.html in your browser."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
