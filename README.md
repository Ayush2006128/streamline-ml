# streamlineML

streamlineML is a user-friendly Streamlit application for building, training, and exporting simple machine learning models on tabular data. It is designed for rapid prototyping and educational purposes.

## Features

- Upload CSV, Parquet, JSON, or XLSX files
- Preview and preprocess data (handle nulls, view stats)
- Configure and train dense neural network models (classification or regression)
- Real-time training progress visualization
- Download trained models in Keras format

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ayush2006128/streamline-ml.git
   cd streamline-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Upload Data**: Use the UI to upload your tabular data file.
2. **Preview & Preprocess**: Inspect your data, handle missing values as needed.
3. **Train Model**: Select features, target, and model/training parameters. Start training and monitor progress.
4. **Download Model**: Once training is complete, download your trained model.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Terms & Privacy

See [TERMS_OF_USE.md](TERMS_OF_USE.md) and [PRIVACY_POLICY.md](PRIVACY_POLICY.md) for more information.