# Experiment Setup and Execution Guide

This guide provides detailed instructions for setting up and running the experiments.

## 1. Downloading Word Vectors and Creating Granular Balls

- **Download Counter-fitted or GloVe Word Vectors**:  
  Download the necessary word vectors from the following sources:  
  - [Counter-fitted Word Vectors](https://www.dropbox.com/scl/fo/gm0bgjwu7l125914ez67o/AI8-VXFzoPEXisJLWtYjHd4?rlkey=op17equzvwgizop2ldmnjggxg&st=kuieauvn&dl=0)

- **Create Granular-balls**:  
  After downloading the word vectors, run the `create_matrix.py` script to generate the granular balls:
  ```bash
  python ./Attacker/create_matrix.py
  ```
  Alternatively, you can directly download the pre-generated granular balls from:  
  - [Pre-generated Granular Balls](https://www.dropbox.com/scl/fo/cykjqdfyrzfm2p8uw48tm/ACytMjF-oe9ZFoLuHwqrMmo?rlkey=8jlmpq8mkrr1fl3jrtid1alsc&st=e3tyn71w&dl=0)

## 2. Training Models

- **Train Models**:  
  You can train the models by running the Python scripts under the `TRAIN` directory:
  ```bash
  python ./TRAIN/your_training_script.py
  ```
  Alternatively, you can download pre-trained model parameters from:  
  - [Pre-trained Models](https://www.dropbox.com/scl/fo/394oj55p2yyljf89ar1bp/AMDmIkWNPe093O_2Z9WmFsQ?rlkey=9g7sp449km4v39clugno3gyde&st=zaa830iq&dl=0)  
  Place the downloaded model in the `./target_models/xxx` directory.

- **Download Datasets**:  
  We have prepared 1000 samples for evaluation in each dataset, which you can download from:  
  - [Evaluation Datasets](https://www.dropbox.com/scl/fo/8swce91geey9gpn689zq2/AACgXzDEITuEceUjP16SDH4?rlkey=37bqe8nflgvq71dz4xgxrrat1&st=a29i2zqe&dl=0)

## 3. Downloading Required Models

- **Download USE Model**:  
  Download the Universal Sentence Encoder model from one of the following sources:
  - [Dropbox USE Model](https://www.dropbox.com/scl/fo/r2pqft97drrnu87i6qjci/AFeaxIJHXPaEIvTizEQKcCA?rlkey=rxi2ks7zi9casia1hc6xej3z2&st=ob77xvau&dl=0)
  - [Google USE Model](https://tfhub.dev/google/universal-sentence-encoder-large/5)  
  Place the downloaded model in the `./Tensorflow` directory.

- **Download GPT Model**:  
  Download the GPT model from:  
  - [GPT Model](dropbox/gpt_model)  
  Place the downloaded model in the `./GPT2` directory.

## 4. Running the Experiment

- **Execute Experiment Code**:  
  You can run the experiment using the `Run.ipynb` notebook.
