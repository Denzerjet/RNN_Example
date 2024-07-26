import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Sample text from Shakespeare
text = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer..."

# Create a character-level mapping
chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Convert text to indices
text_indices = [char_to_idx[char] for char in text]

# Hyperparameters
seq_length = 10  # Length of input sequence
batch_size = 1  # Batch size
hidden_size = 256 
num_layers = 3  # Number of RNN layers
learning_rate = 0.001
num_epochs = 120

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.reshape(-1, hidden_size))  # Process all time steps
        return out, hidden
    
    def init_hidden(self):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


input_size = len(chars)
output_size = len(chars)

model = RNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    hidden = model.init_hidden()
    loss_total = 0

    for i in range(0, len(text_indices) - seq_length, seq_length):
        inputs = text_indices[i:i+seq_length]
        targets = text_indices[i+1:i+seq_length+1]

        inputs = torch.tensor(inputs, dtype=torch.long).view(batch_size, -1)
        targets = torch.tensor(targets, dtype=torch.long).view(batch_size, -1)

        inputs_one_hot = torch.zeros(seq_length, batch_size, input_size)
        for b in range(batch_size):
            for j in range(seq_length):
                inputs_one_hot[j, b, inputs[b, j]] = 1

        inputs_one_hot = inputs_one_hot.permute(1, 0, 2)

        outputs, hidden = model(inputs_one_hot, hidden.detach())
        
        # Reshape outputs to (batch_size*seq_length, output_size) and targets to (batch_size*seq_length)
        outputs = outputs.view(batch_size * seq_length, output_size)
        targets = targets.view(batch_size * seq_length)
        
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_total:.4f}')

def generate_text(model, start_text, length):
    model.eval()
    generated_text = start_text
    input_text = start_text
    hidden = model.init_hidden()

    for _ in range(length):
        inputs = torch.tensor([char_to_idx[char] for char in input_text], dtype=torch.long).view(batch_size, -1)
        inputs_one_hot = torch.zeros(len(input_text), batch_size, input_size)
        for b in range(batch_size):
            for j in range(len(input_text)):
                inputs_one_hot[j, b, inputs[b, j]] = 1

        inputs_one_hot = inputs_one_hot.permute(1, 0, 2)

        outputs, hidden = model(inputs_one_hot, hidden)
        
        # Get the last time step output
        last_output = outputs[-1].view(1, -1)  # Reshape to (1, output_size)
        _, predicted_idx = torch.max(last_output, 1)

        next_char = idx_to_char[predicted_idx.item()]
        generated_text += next_char
        input_text = generated_text[-seq_length:]

    return generated_text

# Generate text
start_text = "To be, or "
print(generate_text(model, start_text, 100))
