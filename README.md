# chordgen

[API](https://chordgen-uxfa.onrender.com/docs) for a generative, transformer based model, to create chord progressions.

An actual implementation of API see [here](https://github.com/akzmuk/genbient) 

## Functional endpoints

- `/start_new/`: returns a random chord from model's vocabulary. Can be used as an adjective for a chord progression.
- `/generate_progression_with_seed/`: returns a generates progression with custom seed. Notice! All availible chords can be found in `./app/chord_vocab.txt`, use them to define your seed.
- `/generate_progression_without_seed/`: returns a generated progression without seed.

## How to use it locally

You can use it either as an inference script or an API.

1. Clone this repository: `git clone https://github.com/akzmuk/chordgen.git`
2. Go to the cloned repo directory
3. Build the Docker Image (It may take a while): `docker build -t chordgen .`
4. Start the Docker Container: `docker run -d --name chordgen_ -p 80:80 chordgen`
5. Check the API docs: http://127.0.0.1/docs

## Acknowledgements

The model was trained on [this data](https://huggingface.co/datasets/ailsntua/Chordonomicon). Huge appreciation to the authors:

```
@article{kantarelis2024chordonomicon,
title={CHORDONOMICON: A Dataset of 666,000 Songs and their Chord Progressions},
author={Kantarelis, Spyridon and Thomas, Konstantinos and Lyberatos, Vassilis and Dervakos, Edmund and Stamou, Giorgos},
journal={arXiv preprint arXiv:2410.22046},
year={2024}
}
```
