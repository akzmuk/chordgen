from app.inference import ChordVocabulary, MemoryEfficientChordTransformer, generate_chord_progression
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import random
import polars

app = FastAPI()

class GenerateWithSeedRequest(BaseModel):
    seed_progression: str
    num_chords: int = 4
    temperature: float = 1.0
    glue_chords: bool = False

class GenerateWithoutSeedRequest(BaseModel):
    num_chords: int = 4
    temperature: float = 1.0

# Load vocabulary and model once when the app starts
vocab = ChordVocabulary.load('app/chord_vocab.txt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MemoryEfficientChordTransformer(
    vocab_size=vocab.size,
    d_model=32,
    nhead=8,
    num_layers=4,
    dim_feedforward=64,
    dropout=0.2
).to(device)

# Load model weights
checkpoint = torch.load('app/chord_transformer.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Load chord mappings from CSV using Polars
chord_mapping_df = polars.read_csv('app/chords_mapping.csv')

# Create a mapping from chord signature (second column) to the list of ints (fourth column)
chord_to_degree = {row[1]: row[3] for row in chord_mapping_df.iter_rows()}

def transform_chord(chord: str) -> str:
    """Transform a slash chord into a regular chord.
    
    Args:
        chord (str): The chord to transform.

    Returns:
        str: The transformed chord.
    """
    return chord.split('/')[0]  # Return the part before the slash

@app.post("/generate_progression_with_seed/")
def generate_with_seed(request: GenerateWithSeedRequest):
    """Generate a chord progression starting with provided seed.
    
    Args:\n
        seed_progression (str): The seed progression (space-separated chord signatures).\n
        num_chords (int): The number of chords to generate.\n
        temperature (float): The sampling temperature for the generation (higher = more random, lower = more deterministic).\n
        glue_chords (bool): Whether to include the seed progression in the result (True) or return only the generated chords (False).\n

    Returns:\n
        dict: The seed + generated chord progression and its degrees (a binary 12-semitone list representation for each chord, commencing with the note C).
    """
    # Parse seed progression
    seed_progression = request.seed_progression.strip().split()
    
    # Check if all seed chords are in vocabulary
    unknown_chords = [c for c in seed_progression if c not in vocab.chord_to_idx]
    if unknown_chords:
        raise HTTPException(status_code=400, detail=f"Unknown chords: {', '.join(unknown_chords)}")
    
    # Generate chord progression
    generated_progression = generate_chord_progression(
        model=model,
        seed_progression=seed_progression,
        vocab=vocab,
        num_chords=request.num_chords,
        temperature=request.temperature,
        device=device
    )
    
    # If glue_chords is False, remove the seed progression from the result
    if not request.glue_chords:
        generated_progression = generated_progression[len(seed_progression):]
    
    # Transform chords to remove slash chords
    transformed_progression = [transform_chord(chord) for chord in generated_progression]
    
    # Get degrees for each transformed chord
    degrees = [chord_to_degree.get(chord, "Unknown") for chord in transformed_progression]
    
    return {
        "generated_progression": transformed_progression,
        "degrees": degrees
    }

@app.post("/generate_progression_without_seed/")
def generate_without_seed(request: GenerateWithoutSeedRequest):
    """Generate a chord progression starting with a random chord.
    
    Args:\n
        num_chords (int): The number of chords to generate.\n
        temperature (float): The sampling temperature for the generation (higher = more random, lower = more deterministic).

    Returns:\n
        dict: The generated chord progression and its degrees (a binary 12-semitone list representation for each chord, commencing with the note C).
    """
    
    # Select a random chord from the vocabulary
    random_seed_chord = random.choice(list(vocab.chord_to_idx.keys()))

    # Ensure we don't select special tokens
    while random_seed_chord in ['<UNK>', '<PAD>']:
        random_seed_chord = random.choice(list(vocab.chord_to_idx.keys()))
    
    # Generate chord progression
    generated_progression = generate_chord_progression(
        model=model,
        seed_progression=[random_seed_chord],  # Use the random chord as the seed
        vocab=vocab,
        num_chords=request.num_chords - 1,  # Generate num_chords - 1 additional chords
        temperature=request.temperature,
        device=device
    )
    
    # Transform chords to remove slash chords
    transformed_progression = [transform_chord(chord) for chord in generated_progression]
    
    # Get degrees for each transformed chord
    degrees = [chord_to_degree.get(chord, "Unknown") for chord in transformed_progression]
    
    return {
        "generated_progression": transformed_progression,
        "degrees": degrees
    }

@app.get("/start_new/")
def start_new():
    """Return a random chord from the vocabulary.
    
    Returns:\n
        dict: The random chord and its degree (a binary 12-semitone list representation for the chord, commencing with the note C).
    """
    
    # Select a random chord from the vocabulary
    random_chord = random.choice(list(vocab.chord_to_idx.keys()))

    # Ensure we don't select special tokens
    while random_chord in ['<UNK>', '<PAD>']:
        random_chord = random.choice(list(vocab.chord_to_idx.keys()))
    
    # Transform the random chord to remove slash chords
    transformed_chord = transform_chord(random_chord)
    
    # Get the degree for the transformed random chord
    degrees = chord_to_degree.get(transformed_chord, "Unknown")
    
    return {
        "chord": transformed_chord,
        "degrees": degrees
    }

@app.get("/")
def home():
    return {"Health check": "OK"}