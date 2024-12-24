
# Image Enhancement Challenge

This repository is part of the **Image Enhancement Challenge** and demonstrates solutions for **Challenge 1: Foundation Enhancement** and **Challenge 2: Background Integration**. This project is designed to optimize eCommerce product images by leveraging advanced image processing techniques.

Future challenges will be implemented to enhance functionality further.

---

## Challenges Implemented

### Challenge 1: Foundation Enhancement
Focused on improving the foundational aspects of product images.

#### Features:
- Adjust brightness and contrast for enhanced image clarity.
- Shadow addition for a more realistic appearance.
- Alignment to ensure the product fills 85% of the frame.

---

### Challenge 2: Background Integration
Enhanced product images with diverse background styles.

#### Features:
- Generated four background styles:
  - **Solid Color**: Harmonized with the product's dominant color.
  - **Gradient**: Smooth gradients tailored for visual appeal.
  - **Studio Setting**: Professional lighting with vignette effects.
  - **Simple Lifestyle Context**: Textured lifestyle-oriented backgrounds.
- Realistic shadows and reflections for product harmonization.
- Consistent lighting across variations.

---

## Future Challenges

The project will be extended with the following challenges:
- **Challenge 3: Text and Banner Integration**: Adding promotional text and banners.
- **Challenge 4: Lifestyle Context Creation**: Creating lifestyle imagery with props and human elements.
- **Challenge 5: Advanced Composition**: Building hero images optimized for eCommerce platforms.

---

## Installation and Setup

### Prerequisites
1. Python 3.8 or higher.
2. Install dependencies listed in `requirements.txt`.
3. Download **YOLOv8** weights (`yolov8s.pt`) and place them in the working directory.
4. Set up an OpenAI API key in a `.env` file.

### Clone the Repository
```bash
git clone https://github.com/your-repo/ImageEnhancementChallenge.git
cd ImageEnhancementChallenge
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Backend
Start the FastAPI server:
```bash
uvicorn backend.app:app --reload
```

### Run the Frontend
Launch the Streamlit application:
```bash
streamlit run frontend/app.py
```

---

## Usage

1. Access the Streamlit interface by running the above command.
2. Upload an image in JPG or PNG format.
3. Select a challenge from the sidebar:
   - **Challenge 1: Foundation Enhancement**
   - **Challenge 2: Background Integration**
4. Click **Process Image** to generate results.
5. View or download the processed image and generated backgrounds.

---

## Directory Structure

```
ImageEnhancementChallenge/
├── backend/
│   ├── challenge_1.py        # Challenge 1 logic
│   ├── challenge_2.py        # Challenge 2 logic
│   ├── utils.py              # Utility functions
│   ├── app.py                # FastAPI backend
├── frontend/
│   ├── app.py                # Streamlit frontend
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .env                      # OpenAI API key
├── static/                   # Stores generated images
└── templates/                # Optional HTML templates
```

---

## How to Contribute

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## Contact

For any queries, feel free to reach out to **Uday Shankar Gattu** and email be at **udaygattu9949@gmail.com**.

---

