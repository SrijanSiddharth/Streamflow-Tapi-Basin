#!/bin/bash
echo "🔧 Starting setup..."

# Install Java
echo "☕ Installing Java..."
bash setup_java.sh

# Install Python dependencies
if [ -f requirements.txt ]; then
    echo "🐍 Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping Python setup."
fi

echo "✅ Setup complete!"