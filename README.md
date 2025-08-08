# Neurosymbolic Program Generation for Visual Question Answering

This repository contains a set of exercises designed to explore different learning strategies for generating symbolic programs to solve a Visual Question Answering (VQA) task. The core idea is to bridge **neural** and **symbolic** processing, leading to more interpretable and generalizable AI systems.

## Project Overview

The project focuses on a VQA task using the CLEVR dataset. The primary goal is not to process the images directly, but to generate a sequence of symbolic program tokens based on a natural language question. These programs then represent the reasoning steps to arrive at the correct answer when executed against a scene graph.

### The Problem: Visual Question Answering (VQA)

VQA requires a model to understand both an image and a question about that image to produce an accurate answer. In a neurosymbolic approach, this is often broken down into two parts:
1.  **Scene Parsing (Neural):** A neural network processes the image to create a symbolic representation, or "scene graph." (This part is pre-processed and provided in this project).
2.  **Program Generation (Neurosymbolic):** A seq2seq model generates a program (a sequence of symbolic functions) based on the input question. This program acts as the reasoning engine.

This project focuses on the second part: **program generation.**

## Learning Objectives

-   **Dataset Preprocessing:** Learn to create and preprocess a dataset for program generation, converting natural language questions and symbolic programs into a format suitable for neural network training.
-   **Seq2Seq Model Implementation:** Implement and understand the architecture of seq2seq models for program generation, using both LSTM and Transformer architectures.
-   **Training Paradigms:** Explore and implement three distinct learning strategies for training the seq2seq models:
    -   **Supervised Learning:** Train the model using ground-truth programs as direct supervision.
    -   **Reinforcement Learning (REINFORCE):** Fine-tune a pre-trained supervised model by rewarding it for generating programs that produce correct answers.
    -   **In-Context Learning (ICL):** Use a Large Language Model (LLM) with carefully crafted prompts and examples to generate programs without explicit model training.
-   **Evaluation:** Analyze the strengths and weaknesses of each approach by evaluating the accuracy of the generated programs.

## The CLEVR Dataset

The project uses a subset of the [CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/). This dataset is ideal for neurosymbolic VQA because it comes with structured scene graphs and ground-truth programs for each question.

-   `CLEVR_Dataset.zip`: This archive contains the necessary scene and question data.

### Data Structure

The dataset contains:
-   **Questions:** Each question is a dictionary containing the natural language query, the corresponding image index, the ground-truth program (a sequence of function calls), and the final answer.
-   **Scenes:** Each scene is a JSON object describing the image's contents, including objects, their attributes (color, size, shape, material), and their relationships.

## Implementation Details

The core of the project is implemented in a series of Python scripts and Jupyter notebooks. The key components include:

### Preprocessing and Dataset Creation

The `neurosymbolic.ipynb` notebook contains code to:
-   Download and unzip the CLEVR dataset.
-   Inspect the raw JSON data for questions, programs, and scenes.
-   Use the `utils/preprocess_questions.py` script to vectorize the questions and programs, converting them into numerical tensors.
-   Create PyTorch `Dataset` and `DataLoader` objects for both training and testing.

### Model Architectures

Two seq2seq models are implemented for program generation:

1.  **LSTM-based Seq2Seq:** A standard encoder-decoder architecture with a bidirectional LSTM encoder and an attention-based LSTM decoder.
2.  **Transformer-based Seq2Seq:** An encoder-decoder architecture based on the "Attention Is All You Need" paper, utilizing self-attention and positional encodings.

Both models are designed to take a vectorized question as input and produce a vectorized program as output.

### Training Strategies

1.  **Supervised Training:**
    -   The models are trained using **teacher forcing**, where the ground-truth program is fed to the decoder at each time step.
    -   The loss function is `CrossEntropyLoss` to predict the next correct token in the program sequence.

2.  **Reinforcement Learning (REINFORCE):**
    -   A `TrainerReinforce` class implements the REINFORCE algorithm.
    -   It uses a pre-trained supervised model as a starting point.
    -   The model samples programs from its policy distribution.
    -   A **reward** of +1 is given for a correct final answer, and 0 otherwise.
    -   A **baseline** (moving average of rewards) is used to calculate the **advantage** to reduce variance.
    -   The policy is updated based on the advantage-weighted log probabilities of the sampled programs.

3.  **In-Context Learning (ICL) with LLMs:**
    -   This approach uses a pre-trained Large Language Model (in this case, TinyLlama for a lightweight example).
    -   A prompt is designed with a **system message** defining the task and several **few-shot examples** (question-program pairs).
    -   The model is then given a new question and asked to complete the prompt by generating the corresponding program.
    -   The number of few-shot examples is varied to observe its effect on the model's performance.

### Evaluation

The performance of each strategy is evaluated using two metrics:

1.  **Executor Accuracy:** The generated program is executed using the `utils/clevr_executor.py` environment. The final answer from the executor is compared with the ground-truth answer from the dataset.
2.  **Program Similarity:** The generated program string is compared directly to the ground-truth program string using metrics like **Exact Match** and **BLEU score**.

The project notebooks provide code to plot and visualize the training loss, reward, and validation accuracy for each approach.

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
