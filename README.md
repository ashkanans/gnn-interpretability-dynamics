---

# **GNN Interpretability Dynamics**

## **Project Overview**

Graph Neural Networks (GNNs) have demonstrated exceptional performance in graph-based tasks. However, their complex
decision-making processes often lack interpretability, making it challenging to trust their outputs in critical
applications. This project investigates the dynamics of GNN interpretability during training and aims to develop tools
and frameworks to uncover insights into the internal representations of GNNs.

### **Goals**

1. Implement methods to evaluate GNN interpretability metrics at different training stages.
2. Explore the trade-offs between accuracy and interpretability using benchmark datasets like MUTAG and IMDB-Binary.
3. Develop frameworks for neuron-level concept detection, global model explanations, and visualization tools.
4. Provide an extensible and modular framework for research in GNN interpretability.

---

## **Project Structure**

```plaintext
gnn-interpretability-dynamics/
├── data/                       # Data and preprocessing
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Preprocessed datasets
│   ├── preprocess.py           # Dataset preprocessing scripts
│   └── download_datasets.py    # Scripts to download datasets
├── models/                     # GNN model implementations
│   ├── __init__.py             # Module initializer
│   ├── base_model.py           # Base class for GNN models
│   ├── gin.py                  # Implementation of GIN
│   ├── gcn.py                  # Implementation of GCN
│   └── layers.py               # Custom GNN layers if needed
├── interpretability/           # Concept extraction and analysis
│   ├── __init__.py             # Module initializer
│   ├── concept_extraction.py   # Concept extraction framework
│   ├── activation_hooks.py     # Tools for capturing neuron activations
├── experiments/                # Scripts for running experiments
│   ├── __init__.py             # Module initializer
│   ├── train_model.py          # Model training scripts
│   └── analyze_interpretability.py # Analyze interpretability-accuracy trade-off
├── notebooks/                  # Jupyter notebooks for tutorials and demos
├── tests/                      # Unit tests for the project
│   ├── __init__.py             # Module initializer
│   ├── test_concept_extraction.py # Tests for concept extraction framework
│   └── test_activation_hooks.py   # Tests for activation tracking
├── utils/                      # Utility scripts
│   ├── __init__.py             # Module initializer
│   ├── logging.py              # Logging configuration
│   ├── config.py               # Configuration management
│   └── plot_helpers.py         # Common plotting utilities
├── requirements.txt            # Python dependencies
├── README.md                   # Project description and usage
├── .gitignore                  # Git ignore rules
└── LICENSE                     # License information
```

---

## **Modules Implemented**

### **1. Data Module**

**Path:** `data/`

- **Responsibility:**  
  Handles dataset download, preprocessing, and organization. Supports both raw and processed datasets for
  reproducibility.
- **Key Scripts:**
    - `download_datasets.py`: Automates downloading datasets like MUTAG and IMDB-Binary.
    - `preprocess.py`: Processes raw datasets into train, validation, and test splits.

---

### **2. Models Module**

**Path:** `models/`

- **Responsibility:**  
  Contains implementations of GNN architectures and custom layers for experimentation.
- **Key Scripts:**
    - `base_model.py`: Provides a shared structure for all GNN models, enabling easy extension.
    - `gin.py`: Implements the Graph Isomorphism Network (GIN).
    - `gcn.py`: Implements the Graph Convolutional Network (GCN).

---

### **3. Interpretability Module**

**Path:** `interpretability/`

- **Responsibility:**  
  Provides tools to understand and analyze neuron-level and model-level interpretability.
- **Key Scripts:**
    - `activation_hooks.py`: Captures neuron activations during forward passes of GNNs.
    - `concept_extraction.py`: Extracts neuron concepts using base concepts and compositional concepts via beam search.
      Evaluates interpretability using Intersection over Union (IoU).

---

### **4. Testing Module**

**Path:** `tests/`

- **Responsibility:**  
  Ensures correctness and robustness of the implemented features.
- **Key Tests:**
    - `test_activation_hooks.py`: Tests the activation tracking system.
    - `test_concept_extraction.py`: Tests the concept extraction framework.

---

