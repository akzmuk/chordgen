import torch
import argparse
import os
from collections import Counter

class ChordVocabulary:
    def __init__(self):
        self.chord_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_chord = {0: '<PAD>', 1: '<UNK>'}
        self.size = 2  # Start with PAD and UNK tokens

    def add_chord(self, chord):
        if chord not in self.chord_to_idx:
            self.chord_to_idx[chord] = self.size
            self.idx_to_chord[self.size] = chord
            self.size += 1

    def chord_to_index(self, chord):
        return self.chord_to_idx.get(chord, self.chord_to_idx['<UNK>'])

    def index_to_chord(self, idx):
        return self.idx_to_chord.get(idx, '<UNK>')

    def save(self, path):
        """Save vocabulary to file"""
        with open(path, 'w') as f:
            for chord, idx in self.chord_to_idx.items():
                f.write(f"{chord}\t{idx}\n")

    @classmethod
    def load(cls, path):
        """Load vocabulary from file"""
        vocab = cls()
        with open(path, 'r') as f:
            for line in f:
                chord, idx = line.strip().split('\t')
                vocab.chord_to_idx[chord] = int(idx)
                vocab.idx_to_chord[int(idx)] = chord
        vocab.size = len(vocab.chord_to_idx)
        return vocab


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Pre-compute positional encodings to save memory during forward pass
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MemoryEfficientChordTransformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.2):
        super(MemoryEfficientChordTransformer, self).__init__()

        # Use smaller embedding dimension
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers with gradient checkpointing
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Final output layer
        self.output_layer = torch.nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.use_checkpointing = True

    def forward(self, src, src_mask=None):
        # Generate padding mask (1 for padding positions, 0 for non-padding)
        src_key_padding_mask = (src == 0)

        # Embedding and positional encoding
        src = self.embedding(src) * (self.d_model ** 0.5)
        src = self.positional_encoding(src)

        # Pass through transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Take the last non-padded token's representation
        batch_indices = torch.arange(output.size(0), device=output.device)
        if src_key_padding_mask is not None:
            # Calculate the position of the last non-padding token
            seq_lengths = src_key_padding_mask.long().eq(0).sum(dim=1) - 1
            last_positions = torch.clamp(seq_lengths, min=0)
        else:
            # If no padding, take the last position
            last_positions = torch.ones(output.size(0), device=output.device) * (output.size(1) - 1)
            last_positions = last_positions.long()

        # Extract the output for the last token position
        last_token_output = output[batch_indices, last_positions]

        # Predict next chord
        logits = self.output_layer(last_token_output)

        return logits


def generate_chord_progression(model, seed_progression, vocab, num_chords=8, temperature=1.0, device=None):
    """
    Generate a chord progression starting with a seed progression
    
    Args:
        model: The trained transformer model
        seed_progression: List of initial chords to start with
        vocab: ChordVocabulary object for mapping between chords and indices
        num_chords: Number of new chords to generate (int)
        temperature: Controls randomness of predictions (higher=more random)
        device: Device to run generation on (cuda or cpu)
        
    Returns:
        List of chords including seed_progression + generated chords
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.use_checkpointing = False  # Disable checkpointing for generation

    # Convert seed progression to indices
    seed_indices = [vocab.chord_to_index(chord) for chord in seed_progression]
    generated_progression = seed_progression.copy()

    # Get sequence length (assuming model was trained with sequence_length=6)
    sequence_length = 6

    # Generate new chords
    with torch.no_grad():
        for _ in range(num_chords):
            # Prepare input sequence for the model
            input_seq = seed_indices[-sequence_length:]  # Use last sequence_length chords as context
            if len(input_seq) < sequence_length:
                input_seq = [0] * (sequence_length - len(input_seq)) + input_seq  # Pad if needed

            input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

            # Get model prediction
            output = model(input_tensor)

            # Apply temperature
            if temperature != 1.0:
                output = output / temperature

            # Filter out special tokens (<UNK> and <PAD>)
            probs = torch.softmax(output, dim=1)
            probs[:, 0] = 0  # Zero probability for <PAD>
            probs[:, 1] = 0  # Zero probability for <UNK>
            
            # Renormalize probabilities
            probs = probs / probs.sum(dim=1, keepdim=True)
            
            # Sample next chord
            next_chord_idx = torch.multinomial(probs, 1).item()

            # Add new chord to progression
            next_chord = vocab.index_to_chord(next_chord_idx)
            generated_progression.append(next_chord)
            seed_indices.append(next_chord_idx)

    return generated_progression


def main():
    parser = argparse.ArgumentParser(description='Generate chord progressions using a trained model')
    parser.add_argument('--model_path', type=str, default='chord_transformer.pth',
                        help='Path to the trained model file')
    parser.add_argument('--vocab_path', type=str, default='chord_vocab.txt',
                        help='Path to the vocabulary file')
    parser.add_argument('--seed', type=str, default='C G Amin F',
                        help='Seed chord progression (space-separated chord signatures)')
    parser.add_argument('--num_chords', type=int, default=4,
                        help='Number of new chords to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random, lower = more deterministic)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cpu') if args.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model and vocabulary files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    if not os.path.exists(args.vocab_path):
        print(f"Error: Vocabulary file not found at {args.vocab_path}")
        return
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab_path}...")
    vocab = ChordVocabulary.load(args.vocab_path)
    print(f"Loaded vocabulary with {vocab.size} chords")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = MemoryEfficientChordTransformer(
        vocab_size=vocab.size,
        d_model=32,
        nhead=8,
        num_layers=4,
        dim_feedforward=64,
        dropout=0.2
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # Parse seed progression
    seed_progression = args.seed.strip().split()
    
    # Check if all seed chords are in vocabulary
    unknown_chords = [c for c in seed_progression if c not in vocab.chord_to_idx]
    if unknown_chords:
        print(f"Warning: The following chords are not in the vocabulary and will be treated as <UNK>: {', '.join(unknown_chords)}")
    
    # Generate chord progression
    print(f"\nGenerating {args.num_chords} new chords with temperature {args.temperature}...")
    generated_progression = generate_chord_progression(
        model=model,
        seed_progression=seed_progression,
        vocab=vocab,
        num_chords=args.num_chords,
        temperature=args.temperature,
        device=device
    )
    
    # Display results
    print("\nSeed progression:", " ".join(seed_progression))
    print("Generated progression:", " ".join(generated_progression))
    
    # Format to clearly show which chords were generated
    new_chords = generated_progression[len(seed_progression):]
    print("\nSeed:", " ".join(seed_progression))
    print("New:", " ".join(new_chords))
    print("Full:", " ".join(generated_progression))


if __name__ == "__main__":
    main()