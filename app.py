from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import product
import random
import os
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Konstanta ===
K = 7
NUCLEOTIDES = ['A', 'T', 'C', 'G']
WINDOW_SIZE = 4000  # 4kb sliding window
STEP_SIZE = 2000     # 2kb step size (50% overlap)

# === Model Definition ===
class KmerClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden1=1024, hidden2=128, dropout=0.38365961722627806):
        super(KmerClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_size),
        )

    def forward(self, x):
        return self.model(x)

# === IUPAC Mapping ===
IUPAC_MAP = {
    'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
    'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'], 'W': ['A', 'T'],
    'K': ['G', 'T'], 'M': ['A', 'C'], 'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'N': ['A', 'C', 'G', 'T']
}

def resolve_ambiguous_bases(seq):
    """Ganti semua huruf ambigu dengan satu huruf acak dari opsi IUPAC-nya."""
    return ''.join(random.choice(IUPAC_MAP.get(base.upper(), ['A'])) for base in seq)

def reverse_complement(seq: str) -> str:
    complement = str.maketrans('ATCG', 'TAGC')
    return seq.translate(complement)[::-1]

def canonical_kmer(kmer: str) -> str:
    rev_kmer = reverse_complement(kmer)
    return min(kmer, rev_kmer)

def generate_all_canonical_kmers(k: int):
    all_kmers = {canonical_kmer(''.join(p)) for p in product(NUCLEOTIDES, repeat=k)}
    return sorted(all_kmers)

# Precompute canonical k-mers
CANONICAL_KMER_ORDER = generate_all_canonical_kmers(K)

def compute_canonical_kmer_freq_vector(seq: str, k: int = K):
    freq = defaultdict(int)
    rev_seq = reverse_complement(seq)

    for s in [seq, rev_seq]:
        for i in range(len(s) - k + 1):
            kmer = s[i:i+k]
            if all(nuc in NUCLEOTIDES for nuc in kmer):
                can_kmer = canonical_kmer(kmer)
                freq[can_kmer] += 1

    return [freq.get(kmer, 0) for kmer in CANONICAL_KMER_ORDER]

def sliding_window_analysis(sequence, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Perform sliding window analysis on sequence and return union of predictions"""
    if len(sequence) <= window_size:
        # If sequence is smaller than window, analyze the whole sequence
        return [sequence]
    
    windows = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        window = sequence[i:i + window_size]
        windows.append(window)
    
    # Add the last window if it's not fully covered
    if len(sequence) % step_size != 0:
        windows.append(sequence[-window_size:])
    
    return windows

def union_predictions(predictions_list):
    """Combine predictions from multiple windows using union (max probability)"""
    if not predictions_list:
        return {}
    
    # Initialize with first prediction
    union_pred = predictions_list[0].copy()
    
    # For each subsequent prediction, take the maximum probability
    for pred in predictions_list[1:]:
        for antibiotic in union_pred:
            # Use maximum probability across all windows
            if pred[antibiotic]['probability'] > union_pred[antibiotic]['probability']:
                union_pred[antibiotic] = pred[antibiotic].copy()
    
    # Recalculate predictions and confidence based on final probabilities
    for antibiotic in union_pred:
        prob = union_pred[antibiotic]['probability']
        threshold = thresholds[label_names.index(antibiotic)]
        
        union_pred[antibiotic]['prediction'] = 'Resistant' if prob > threshold else 'Susceptible'
        union_pred[antibiotic]['confidence'] = float(prob) if prob > threshold else float(1 - prob)
    
    return union_pred

# === Global Variables ===
model = None
thresholds = None
label_names = [
    "aminoglycoside",
    "carbapenem",
    "cephalosporin",
    "fluoroquinolone",
    "lincosamide",
    "macrolide",
    "monobactam",
    "penicillin beta-lactam",
    "peptide",
    "phenicol",
    "tetracycline"
]

def load_model_and_data():
    global model, thresholds
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(CANONICAL_KMER_ORDER)
    output_size = len(label_names)
    
    model = KmerClassifier(input_size, output_size)
    model.load_state_dict(torch.load("./model/final/bok_model.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Load thresholds
    thresholds = np.load("./model/bag-of-kmers/thresholds.npy")
    
    print("Model and data loaded successfully!")

def predict_single_sequence(sequence):
    """Predict resistance for a single sequence"""
    if model is None:
        raise ValueError("Model not loaded")
    
    # Clean ambiguous bases
    cleaned_sequence = resolve_ambiguous_bases(sequence)
    
    # Compute k-mer features
    kmer_features = compute_canonical_kmer_freq_vector(cleaned_sequence)
    
    # Convert to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor([kmer_features], dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(X)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Apply thresholds
    predictions = {}
    for i, (prob, threshold) in enumerate(zip(probs, thresholds)):
        predictions[label_names[i]] = {
            'probability': float(prob),
            'prediction': 'Resistant' if prob > threshold else 'Susceptible',
            'confidence': float(prob) if prob > threshold else float(1 - prob)
        }
    
    return predictions

def predict_resistance(sequence, sequence_type='ARG'):
    """Predict antibiotic resistance from DNA sequence"""
    if sequence_type == 'ARG':
        # Direct analysis for ARG sequences
        return predict_single_sequence(sequence)
    
    elif sequence_type == 'Other':
        # Sliding window analysis for other sequences
        windows = sliding_window_analysis(sequence)
        
        # Get predictions for each window
        window_predictions = []
        for window in windows:
            pred = predict_single_sequence(window)
            window_predictions.append(pred)
        
        # Union the predictions
        final_predictions = union_predictions(window_predictions)
        
        # Add metadata about analysis
        final_predictions['_analysis_info'] = {
            'total_windows': len(windows),
            'window_size': WINDOW_SIZE,
            'step_size': STEP_SIZE,
            'sequence_length': len(sequence)
        }
        
        return final_predictions
    
    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sequence_type = data.get('sequence_type', 'ARG')
        sequence = data.get('sequence', '').strip()
        
        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400
        
        # Make prediction
        results = predict_resistance(sequence, sequence_type)
        
        # Extract analysis info if present
        analysis_info = results.pop('_analysis_info', None)
        
        response_data = {
            'success': True,
            'predictions': results,
            'sequence_type': sequence_type,
            'sequence_length': len(sequence)
        }
        
        if analysis_info:
            response_data['analysis_info'] = analysis_info
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_file', methods=['POST'])
def predict_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        sequence_type = request.form.get('sequence_type', 'ARG')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content
        content = file.read().decode('utf-8')
        
        # Handle FASTA format
        if file.filename.endswith('.fasta') or file.filename.endswith('.fa'):
            lines = content.strip().split('\n')
            sequence = ''
            for line in lines:
                if not line.startswith('>'):
                    sequence += line.strip()
        else:
            # Assume plain text sequence
            sequence = content.strip()
        
        if not sequence:
            return jsonify({'error': 'No valid sequence found in file'}), 400
        
        # Make prediction
        results = predict_resistance(sequence, sequence_type)
        
        # Extract analysis info if present
        analysis_info = results.pop('_analysis_info', None)
        
        response_data = {
            'success': True,
            'predictions': results,
            'sequence_type': sequence_type,
            'sequence_length': len(sequence),
            'filename': file.filename
        }
        
        if analysis_info:
            response_data['analysis_info'] = analysis_info
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model_and_data()
    app.run(debug=True)