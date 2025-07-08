# ARG Resistance Classifier

A machine learning-based antibiotic resistance prediction system using k-mer features for DNA sequence analysis. This project is an improved reimplementation of [amrfad/antibiotic-resistance-prediction](https://github.com/amrfad/arg-resistance-classifier) with enhanced architecture and methodology.

## 📋 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Performance Analysis](#performance-analysis)
- [Installation](#installation)
- [Usage](#usage)

## 🔬 Overview

This project implements a multi-label classification system for predicting antibiotic resistance from ARG (antibiotic resistance gene) sequences. The system uses canonical k-mer features and a neural network architecture to classify sequences into 11 different antibiotic resistance categories.
> ⚠️ **Note**: This project is a **proof-of-concept** intended to demonstrate the feasibility of ARG classification using bag-of-kmers method and neural networks. It is not designed for clinical or diagnostic use.

### Key Features
- **Multi-label Classification**: Predicts resistance to 11 different antibiotic classes
- **Canonical K-mer Features**: Uses 7-mer canonical representation for efficient feature extraction
- **Sliding Window Analysis**: Handles long sequences with overlapping 4kb windows
- **Web Interface**: User-friendly Flask application for easy inference
- **Robust Preprocessing**: Handles ambiguous nucleotides using IUPAC nucleotide codes

## 🏗️ Model Architecture

### Neural Network Design
- **Architecture**: Multi-layer Perceptron (MLP) with 3 layers
- **Layer Configuration**: 
  - Input Layer: Variable size (canonical 7-mer features)
  - Hidden Layer 1: 1,024 neurons
  - Hidden Layer 2: 128 neurons
  - Output Layer: 11 neurons (one per antibiotic class)
- **Activation Functions**: ReLU (hidden layers), Sigmoid (output layer)
- **Regularization**: Dropout (0.384)
- **Loss Function**: BCEWithLogitsLoss with class weighting

### Feature Engineering
- **K-mer Size**: 7-mer for optimal balance between specificity and generalization
- **Canonical Representation**: Reduces feature dimensionality while preserving biological meaning
- **Ambiguous Nucleotide Resolution**: IUPAC-compliant random nucleotide substitution
- **Sequence Segmentation**: 4kb sliding windows with 50% overlap for long sequences

### Optimization
- **Framework**: Optuna with 50 trials
- **Optimization Metric**: Macro F2-score
- **Hyperparameter Tuning**: Learning rate, batch size, hidden layer sizes, dropout rate
- **Cross-validation**: Multilabel Stratified K-Fold for balanced data splits

## 📊 Dataset

### Data Source
- **Database**: [CARD (Comprehensive Antibiotic Resistance Database)](https://card.mcmaster.ca/)
- **Total Sequences**: 6,386 DNA sequences

### Target Classes
The system predicts resistance to 11 antibiotic classes (selected from 43 original labels with ≥100 samples):

- Aminoglycoside antibiotic
- Carbapenem
- Cephalosporin
- Fluoroquinolone antibiotic
- Lincosamide antibiotic
- Macrolide antibiotic
- Monobactam
- Penicillin beta-lactam
- Peptide antibiotic
- Phenicol antibiotic
- Tetracycline antibiotic

### Data Distribution
- **Training Set**: 81% (5,174 samples)
- **Validation Set**: 9% (575 samples)
- **Test Set**: 10% (638 samples)

## 📈 Performance Analysis

### Overall Performance
**Macro F2-score: 0.7575**

The model demonstrates strong performance across different antibiotic classes, with notable variations based on class characteristics and sample availability.

### Performance by Antibiotic Class

#### High-Performance Classes (F2-score > 0.90)
| Antibiotic Class | Precision | Recall | F1-Score | F2-Score | Support |
|------------------|-----------|--------|----------|----------|---------|
| **Cephalosporin** | 0.96 | 0.98 | 0.97 | **0.9760** | 340 |
| **Carbapenem** | 0.95 | 0.97 | 0.96 | **0.9684** | 258 |
| **Monobactam** | 1.00 | 0.95 | 0.98 | **0.9633** | 132 |
| **Penicillin beta-lactam** | 0.96 | 0.96 | 0.96 | **0.9622** | 312 |

*These classes show excellent performance with high precision and recall, benefiting from large sample sizes and distinct sequence patterns.*

#### Moderate-Performance Classes (F2-score 0.70-0.90)
| Antibiotic Class | Precision | Recall | F1-Score | F2-Score | Support |
|------------------|-----------|--------|----------|----------|---------|
| **Fluoroquinolone** | 0.46 | 0.90 | 0.61 | **0.7542** | 30 |
| **Aminoglycoside** | 0.44 | 0.90 | 0.59 | **0.7459** | 30 |
| **Lincosamide** | 0.47 | 0.82 | 0.60 | **0.7143** | 11 |

*These classes demonstrate good recall but lower precision, indicating some classification overlap with other resistance mechanisms.*

#### Lower-Performance Classes (F2-score < 0.70)
| Antibiotic Class | Precision | Recall | F1-Score | F2-Score | Support |
|------------------|-----------|--------|----------|----------|---------|
| **Macrolide** | 0.31 | 0.81 | 0.45 | **0.6115** | 21 |
| **Tetracycline** | 0.46 | 0.63 | 0.53 | **0.5882** | 19 |
| **Peptide** | 0.32 | 0.63 | 0.42 | **0.5263** | 19 |
| **Phenicol** | 0.47 | 0.54 | 0.50 | **0.5224** | 13 |

*These classes face challenges from limited training data and potentially more diverse resistance mechanisms.*

### Key Performance Insights

1. **Sample Size Impact**: Classes with larger sample sizes (>100 samples) consistently show better performance
2. **Beta-lactam Excellence**: Beta-lactam antibiotics (Carbapenem, Cephalosporin, Monobactam, Penicillin) show exceptional performance, likely due to well-characterized resistance mechanisms
3. **Precision-Recall Trade-off**: Some classes prioritize recall over precision, suitable for screening applications where false negatives are more costly
4. **Macro vs. Class-specific**: While macro F2-score is 0.7575, individual class performance varies significantly (0.5224 to 0.9760)

### Aggregate Metrics
- **Macro Average**: Precision 0.62, Recall 0.83, F1 0.69

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Flask 2.0+
- Additional dependencies: scikit-learn, pandas, numpy, iterstrat, optuna

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/amrfadh/arg-resistance-classifier.git
cd arg-resistance-classifier
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio
pip install flask pandas numpy scikit-learn iterstrat optuna
```

4. **Verify installation**
```bash
python -c "import torch; print(torch.__version__)"
```

## 🔧 Usage

### Web Interface

**Launch the Flask application:**
```bash
python app.py
```

Access the application at `http://localhost:5000`

**Web Interface Features:**
- Direct DNA sequence input via text area
- FASTA/Text file upload support
- Analysis type selection (ARG sequence vs. Other sequence)
- Interactive result visualization with confidence scores
- Sliding window analysis for long sequences

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments

- **CARD Database** for providing comprehensive antibiotic resistance data
- **PyTorch Team** for the deep learning framework
- **Optuna** for hyperparameter optimization capabilities
- **scikit-learn** for machine learning utilities

---

*For questions or support, please open an issue on GitHub or contact the maintainers.*