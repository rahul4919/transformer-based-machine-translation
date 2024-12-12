
# Transformer for Machine Translation

This assignment implements a transformer model for machine translation, following the original architecture proposed in "Attention is All You Need." The assignment includes code for training and testing the model, and a pre-trained model can be downloaded from a provided Google Drive link.

---

## Assignment Structure

### Source Code
- **`train.py`**: Main code for training the transformer model on the provided dataset.
  - Trains a transformer model for machine translation and saves the trained model to a file named `transformer.pt`.
  - **Instructions to execute**:
    ```bash
    python train.py
    ```
    Ensure all required dependencies are installed and the training data is properly loaded in the code.

- **`test.py`**: Code for testing the pre-trained model on the test dataset.
  - Loads the model saved during training and evaluates it on the test set.
  - Calculates the BLEU score for each sentence and saves it in `testbleu.txt`.
  - **Instructions to execute**:
    ```bash
    python test.py
    ```
    Ensure that `transformer.pt` is in the correct directory before running this file.

- **`encoder.py`**: Contains the implementation of the encoder class and related functions.
  - Defines the `Encoder` module, positional encoding (either sinusoidal or learnable), multi-head attention mechanism, and other related classes used to build the encoder stack.

- **`decoder.py`**: Contains the implementation of the decoder class and related functions.
  - Implements the `Decoder` module, including self-attention, cross-attention with the encoder output (memory), and other relevant components to decode the target sequence.

- **`utils.py`**: Provides any other helper functions or classes required for data processing, model setup, training, and evaluation.
  - May include functions for data preprocessing, tokenization, batching, and BLEU score computation.

### Pretrained Models
- **`transformer.pt`**: Pre-trained transformer model.
  - You can download the pre-trained model from this [Google Drive link](https://drive.google.com/file/d/1U-ChcaN7xN0kyzqoqrkBmVU2NCbJrirR/view?usp=sharing).
  - **Loading the pretrained model**:
    ```python
    import torch
    from model import Transformer  # Assuming your model class is in model.py
    
    model = Transformer(...)  # Instantiate your model architecture
    model.load_state_dict(torch.load('transformer.pt'))
    model.eval()  # Set model to evaluation mode
    ```
    Ensure that `transformer.pt` is placed in the appropriate directory before attempting to load the model.

### Text Files
- **`testbleu.txt`**: Contains the BLEU scores for each sentence in the test set.
  - Each line is formatted as:
    ```
    <generated_sentence> <bleu_score>
    ```
  - This file is generated when running `test.py` and is saved to the current working directory.

---

## Setup and Installation
1. **Install Dependencies**: Ensure that you have all required libraries installed. You can create a virtual environment and use `pip` to install packages:
   ```bash
   pip install -r requirements.txt
   ```
   If no `requirements.txt` is provided, install necessary packages manually:
   ```bash
   pip install torch sacrebleu
   ```

2. **Download the Pre-trained Model**: If you wish to test the pre-trained model without training, download `transformer.pt` from the [Google Drive link](https://drive.google.com/file/d/1U-ChcaN7xN0kyzqoqrkBmVU2NCbJrirR/view?usp=sharing) and place it in your project directory.

3. **Prepare the Dataset**: Ensure that the training and test datasets are correctly preprocessed and loaded in the codebase. Modify the data paths as needed in `train.py` and `test.py`.

---

## Training the Model
To train the transformer from scratch:
```bash
python train.py
```
This will train the model using the provided dataset and save the trained model to `transformer.pt`.

### Parameters and Configuration
- **Model Hyperparameters**: You can modify model hyperparameters like `d_embed`, `num_heads`, `max_seq_len`, `dropout`, etc., in the configuration section of `train.py`.
- **Training Parameters**: Adjust training parameters such as the number of epochs, learning rate, and batch size directly in `train.py`.

---

## Testing the Model
To test the pre-trained transformer or a newly trained model:
```bash
python test.py
```
This will load `transformer.pt`, evaluate it on the test set, and save the BLEU scores to `testbleu.txt`.

---

## Implementation Assumptions
1. **Positional Encoding**: By default, the code uses learnable positional embeddings. If required, you can switch to fixed sinusoidal encodings by modifying `encoder.py` and `decoder.py`.
2. **BLEU Score Calculation**: The BLEU score is computed for each sentence in the test set using `sacrebleu`.
3. **Data Loading**: The `Dataloaders` class in `train.py` and `test.py` assumes preprocessed and tokenized input data. Make sure to adjust the data paths and processing as needed.

---

## Notes
- **Model Checkpoints**: The best model checkpoint is saved automatically to `transformer.pt` during training based on the lowest validation loss.
- **Testing Output**: The test output includes BLEU scores for individual sentences and an overall BLEU score for the test set, providing detailed insights into model performance.

For further questions or issues, please refer to the code documentation or reach out to the project maintainers.

---
