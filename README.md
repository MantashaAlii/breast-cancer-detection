# Breast Cancer Detection

A Python-based application for detecting breast cancer using machine learning techniques with a user-friendly Streamlit interface.

## Features

- Upload breast scan images and get instant predictions.
- Simple and interactive Streamlit UI.
- Includes unit tests to verify functionality.

## Project Structure


├── app.py         # Streamlit UI for uploading images and making predictions

├── main.py        # Backend logic for model predictions

├── test.py        # Test scripts for validation

├── requirements.txt  # Project dependencies


## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. *Clone the repository:*

   bash
   git clone https://github.com/MantashaAlii/breast-cancer-detection.git
   
   cd Breast-Cancer-Detection
   

3. *Create a virtual environment (recommended):*

   bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   

4. *Install dependencies:*
   bash
   pip install -r requirements.txt
   

## Running the Application

1. *Start the Streamlit app:*

   bash
   streamlit run app.py
   

2. *Access the application:*

   - Open your browser and navigate to http://localhost:8501.

3. *Using the Application:*
   - Click on the *Browse files* button to upload a breast scan image.
   - Click the *Predict* button to get the prediction result.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.
