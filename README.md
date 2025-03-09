# chordgen

API for a generative, Transformer based model, that creates chord progressions.

## How to use it locally

1. Clone this repository: `git clone https://github.com/akzmuk/chordgen.git`
2. Go the the cloned repo directory
3. Build the Docker Image (It may take a while): `docker build -t chordgen .`
4. Start the Docker Container: `docker run -d --name chordgen_ -p 80:80 chordgen`
5. Check the API docs: http://127.0.0.1/docs

Notice! All availible chords can be found in ./app/chord_vocab.txt, use them to define your seed.

## Reference

The model was trained on [this data](https://huggingface.co/datasets/ailsntua/Chordonomicon).

Kantarelis, S., Thomas, K., Lyberatos, V., Dervakos, E., & Stamou, G. (2024). CHORDONOMICON: A dataset of 666,000 songs and their chord progressions. arXiv preprint arXiv:2410.22046. https://arxiv.org/abs/2410.22046
