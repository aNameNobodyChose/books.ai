import argparse
import random
import torch
from train.model import NeuralNet
from train.nltk_utils import bag_of_words, tokenize
from train.train import load_intents

def chatLoop(all_words, model, tags, bot_name, intents):
    while True:
        sentence = input('You: ')
        if sentence == "quit":
            break
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}\n")
        else:
            print(f"{bot_name}: I do not understand...")

def main():
    parser = argparse.ArgumentParser(description="Begin chatting!")
    parser.add_argument("--intents", required=True, help="Path to the intents file")
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--bot-name", required=True, help="Bot name")
    args = parser.parse_args()

    intents = load_intents(args.intents)

    # This part is for loading the trained model
    data = torch.load(args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = args.bot_name
    print("Let's chat! type 'quit' to exit")
    chatLoop(all_words, model, tags, bot_name, intents)


# Run script
if __name__ == "__main__":
    main()