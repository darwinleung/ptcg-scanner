# PTCG Scanner

A Python application that uses computer vision and machine learning to detect and identify Pokémon Trading Card Game (PTCG) cards from images.

## Features

- Card detection using Segment Anything Model (SAM)
- Card identification using CLIP embeddings
- Support for multiple cards in a single image
- Streamlit web interface for easy interaction
- Mobile-friendly design

## Installation

1. Clone the repository:
```bash
git clone https://github.com/darwinleung/ptcg-scanner.git
cd ptcg-scanner
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required model files:
- SAM model: [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- Place the model file in the project root directory

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload an image containing Pokémon cards

4. The app will:
   - Detect individual cards in the image
   - Identify each card using CLIP embeddings
   - Display the results with confidence scores

## Project Structure

- `app.py`: Main Streamlit application
- `sam_utils.py`: Card detection using SAM
- `utils.py`: Utility functions
- `build_index.py`: Script to build CLIP embeddings index
- `data/`: Sample card images
- `photos/`: Test images

## Requirements

- Python 3.8+
- PyTorch
- Segment Anything Model
- CLIP
- Streamlit
- Other dependencies listed in requirements.txt

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 