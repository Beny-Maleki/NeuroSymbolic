# Neurosymbolic Program Generation for Visual Question Answering

This repository contains a set of exercises designed to explore different learning strategies for generating symbolic programs to solve a Visual Question Answering (VQA) task. The core idea is to bridge **neural** and **symbolic** processing, leading to more interpretable and generalizable AI systems. These exercises are based on the [Neural-Symbolic VQA paper](https://arxiv.org/pdf/1810.02338).

## Project Overview

The project focuses on a VQA task using the CLEVR dataset. The primary goal is not to process the images directly, but to generate a sequence of symbolic program tokens based on a natural language question. These programs then represent the reasoning steps to arrive at the correct answer when executed against a scene graph.

## Getting Started

### Prerequisites

-   Python 3.x
-   `torch`
-   `transformers`
-   `tqdm`
-   `nltk`
-   `h5py`
-   `gdown`
-   `ijson`
-   `matplotlib`

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/Beny-Maleki/NeuroSymbolic.git
    cd NeuroSymbolic
    ```
2.  Run the notebook cells sequentially to download the data, preprocess it, train the models, and perform the evaluations. You will need to have a Google Drive mounted to save the pre-trained model weights.
