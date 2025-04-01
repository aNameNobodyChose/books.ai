import argparse
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

class HyperParameters:
    def __init__(
        self,
        batch_size,
        hidden_size,
        X_train,
        tags,
        learning_rate,
        num_epochs,
    ):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = len(tags)
        self.input_size = len(X_train[0])
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

def load_intents(intents_file):
    with open(intents_file, 'r') as f:
        intents = json.load(f)
        return intents

# intents.json contains these fields: tag, pattern, and response
# tag refers to the type of conversation
# pattern: Example types of queries that can be asked
# response: The answers that can be given

def get_training_data(intents):
    all_words = []
    tags = []
    sentences_with_tag = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            sentences_with_tag.append((w, tag))

    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    return all_words, tags, sentences_with_tag

def extract_x_train_y_train(all_words, tags, sentences_with_tag):
    X_train = []
    y_train = []

    for (pattern_sentence, tag) in sentences_with_tag: 
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train) # Training data (Array of array of bag of words)
    y_train = np.array(y_train) # Labels (intents of each sentence)
    return X_train, y_train

def training_loop(
        hyper_parameters,
        all_words,
        tags,
        train_loader,
        device,
        model,
        criterion,
        optimizer
    ):
    for epoch in range(hyper_parameters.num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            #forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            #backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1 ) % 100 == 0:
            print(f'epoch {epoch + 1}/{hyper_parameters.num_epochs}, loss={loss.item():.4f}')

    print(f'final loss, loss={loss.item():.4f}')
    trained_data = {
        "model_state": model.state_dict(),
        "input_size": hyper_parameters.input_size,
        "output_size": hyper_parameters.output_size,
        "hidden_size": hyper_parameters.hidden_size,
        "all_words": all_words,
        "tags": tags
    }
    return trained_data

def save_trained_model(trained_data, output_file):
    torch.save(trained_data, output_file)
    print(f'training complete. File saved to {output_file}')

def main():
    parser = argparse.ArgumentParser(description="Process a story and write training data.")
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    args = parser.parse_args()

    intents = load_intents(args.input)
    all_words, tags, sentences_with_tag = get_training_data(intents)

    # Initialize dataset with training data
    X_train, y_train = extract_x_train_y_train(all_words, tags, sentences_with_tag)
    dataset = ChatDataset(X_train, y_train)

    # Hyperparameters
    hyper_parameters = HyperParameters(
        batch_size=8,
        hidden_size=8,
        X_train=X_train,
        tags=tags,
        learning_rate=0.001,
        num_epochs=1000
    )

    train_loader = DataLoader(dataset=dataset, batch_size=hyper_parameters.batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(hyper_parameters.input_size, hyper_parameters.hidden_size, hyper_parameters.output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameters.learning_rate)

    trained_data = training_loop(
        hyper_parameters,
        all_words,
        tags,
        train_loader,
        device,
        model,
        criterion,
        optimizer
    )

    save_trained_model(trained_data, args.output)

# Run script
if __name__ == "__main__":
    main()