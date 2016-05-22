# Random Sentence Generator

Generate random sentences based on an input file using Markov chains.

```
python generate_from_file.py input.txt -c chain_length
```

Chain length is the number of words to look back when generating a new word.
Setting chain length to be too high (> 3) will result in sampling sentences
verbatim out of the text.

The Wizard of Oz text is included for testing. You can generate a sentence
using:

```
python generate_from_file.py wizard_of_oz.txt
```

Generate random sentences using the `Generator` class. (see example of use in
`generate_from_file.py`)

```python
from sentence_generator import Generator
generator = Generator(sentences, chain_length)
print(generator.generate())
```

Requirements:

 * [NLTK](http://www.nltk.org/install.html)
