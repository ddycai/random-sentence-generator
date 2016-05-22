""" Generate random sentences from a corpus using Markov chains."""
import nltk

class Generator:
    # Ignore quotes in sentence generation.
    IGNORED = ['"', '\'']
    # Symbols that should not have space preceding it.
    NO_SPACE_BEFORE = [',', '.', '?', ':', ';', ')', '!', "n't", "''", "'t"]
    NO_SPACE_BEFORE_PREFIX = ['.', '\'']
    NO_SPACE_AFTER = ['(', '``']
    START = '<s>'
    END = '</s>'

    def __init__(self, sentences, chain_len = 2):
        """
        Creates a new sentence generator based on the given corpus.

        Args:
            sentences (list(list(string))): a list of tokenized senteces
            chain_len (int): max number of words to look back when generating a new
            sentence
        """
        self.lm = {} # stores the language model
        self.chain_len = chain_len
        words = self._delimit_sentences(sentences)
        
        self.lm[1] = self._bigram_mle_model(words)
        for i in range(2, chain_len + 1):
            self.lm[i] = self._ngram_mle_model(words, i + 1)

    @staticmethod
    def _bigram_mle_model(words):
        cfdist = nltk.ConditionalFreqDist(nltk.bigrams(words))
        return nltk.ConditionalProbDist(cfdist, nltk.MLEProbDist)

    @staticmethod
    def _ngram_mle_model(words, n):
        ngrams = nltk.ngrams(words, n)
        cfdist = nltk.ConditionalFreqDist((tuple(x[:(n - 1)]), x[n - 1]) for x in \
                ngrams)
        return nltk.ConditionalProbDist(cfdist, nltk.MLEProbDist)

    @staticmethod
    def _delimit_sentences(sents):
        """
        Given a list of sentences returns a list of tokens that delimits where
        sentences start and end with <s> and </s>.

        Args:
            sentences (list(list(string))): a list of tokenized senteces
        """
        result = []
        for sent in sents:
            result.append(Generator.START)
            result.extend([word for word in sent if word not in \
                Generator.IGNORED])
            result.append(Generator.END)
        return result

    @staticmethod
    def stitch(sentence):
        """
        Stitch sentence parts together with proper spacing. For example, for
        punctuations, contractions etc.
        """
        result = []
        stack = []
        buf = ""
        for word in sentence:
            if word in Generator.NO_SPACE_AFTER:
                buf += word
            elif word in Generator.NO_SPACE_BEFORE or word[0] in \
                    Generator.NO_SPACE_BEFORE_PREFIX:
                if len(result) == 0:
                    result.append(buf + word)
                else:
                    result[-1] += buf + word
                buf = ""
            else:
                result.append(buf + word)
                buf = ""
        return " ".join(result)

    def generate(self, as_list = False):
        """
        Generates a random sentence.

        Args:
            as_list (bool): Returns a list of tokens if True, otherwise returns a
            single string.
        """
        # Generate words until we reach end of sentence
        sentence = []
        context = [Generator.START]
        while context[-1] != Generator.END:
            # Special case for bigrams
            if len(context) == 1:
                cur = self.lm[1][context[0]].generate()
                context.append(cur)
            else:
                cur = self.lm[len(context)][tuple(context)].generate()
                context.append(cur)
                if len(context) >= self.chain_len:
                    context.pop(0)

            if cur != Generator.END:
                sentence.append(cur)

        if as_list:
            return sentence
        else:
            return self.stitch(sentence)
