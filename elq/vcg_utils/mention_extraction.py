import nltk

from vcg_utils import utils


class NgramNpParser:

    def __init__(self, exclude_pos=set(), exclude_if_first=set(), exclude_prefix=set(), exclude_suffix=set(), exclude_alone=set()):
        punctuation_tags = {"``", "''", "(", ")", ",", ".", ":", "--"}
        self._exclude_pos = set(exclude_pos)
        self._exclude_if_first = set(exclude_if_first)
        self._exclude_prefix = set(exclude_prefix) | punctuation_tags
        self._exclude_suffix = set(exclude_suffix) | punctuation_tags
        self._exclude_alone = set(exclude_alone) | punctuation_tags

    def parse(self, tagged_text, ngram_len=-1):
        ngrams = []
        if len(tagged_text) == 0:
            return ngrams
        if tagged_text[0]['pos'] in self._exclude_if_first:
            tagged_text = tagged_text[1:]
        if ngram_len == -1:
            for l in range(len(tagged_text), 0, -1):
                ngrams += list(nltk.ngrams(tagged_text, l))
        else:
            ngrams += list(nltk.ngrams(tagged_text, ngram_len))
            ngrams += [n[:-1] for n in ngrams if len(n) > 1 and n[-1]['pos'] in {"NN", "NNS"}]
            ngrams += [n[1:] for n in ngrams if len(n) > 1 and n[0]['pos'] in {"NN", "NNS"}]
        ngrams = [n for n in ngrams
                if len({el[i] for el in n for i in {'pos', 'ner'}} & self._exclude_pos) == 0
                and (len(n) == 1 or (n[0]['pos'] not in self._exclude_prefix
                        and n[0]['word'].lower() not in utils.stop_words_en
                        and n[-1]['pos'] not in self._exclude_suffix
                        and n[-1]['word'].lower() not in utils.stop_words_en)
                    )
                and not(len(n) == 1 and (n[0]['pos'] in self._exclude_alone or n[0]['word'].lower() in utils.stop_words_en))]
        return ngrams


# np_parser = NgramNpParser(exclude_pos={".", "ORDINAL", "TIME", "PERCENT"},
#                           exclude_if_first={"WDT", "WP", "WP$", "WRB", "VBZ", "VB", "VBP"},
#                           exclude_prefix={"IN", "DT", "CC", "POS"},
#                           exclude_suffix={"IN", "DT", "CC"},
#                           exclude_alone={"IN", "DT", "PDT", "POS", "PRP", "PRP$", "CC", "TO",
#                                          "VBZ", "VBD", "VBP"
#                                          })

np_parser = NgramNpParser(
    exclude_pos={},
    exclude_if_first={},
    exclude_prefix={"CC"},
    exclude_suffix={"CC"},
    exclude_alone={"IN", "DT", "PDT", "PRP", "PRP$", "CC", "TO"}
)

def extract_entities(tagged_input, ngram_len=-1, linked_token_ids=set()):
    """
    Extract entities from the NE tags and POS tags of a sentence. Regular nouns are lemmatized to get rid of plurals.

    :param tagged_input: list of tokens as dicts with tags
    :param ngram_len: length of the extracted fragments
    :param linked_token_ids: token ids that are already linked and should be excluded
    :return: list of entities in the order: NE>NNP>NN
    >>> [f['tokens'] for f in extract_entities(utils.get_tagged_from_server("who are the current senators from missouri", caseless=True), ngram_len=3)]
    [['senators', 'from', 'missouri']]
    >>> [f['tokens'] for f in extract_entities(utils.get_tagged_from_server("who are the current senators from missouri", caseless=True), linked_token_ids={3,4})]
    [['missouri']]
    >>> [f['type'] for f in extract_entities(utils.get_tagged_from_server("who was the president after jfk died", caseless=True), ngram_len=3)]
    ['NNP']
    >>> [f['tokens'] for f in extract_entities(utils.get_tagged_from_server("what character did john noble play in lord of the rings?", caseless=True), ngram_len=4)]
    [['character', 'did', 'john', 'noble'], ['noble', 'play', 'in', 'lord'], ['lord', 'of', 'the', 'rings']]
    >>> [f['tokens'] for f in extract_entities(utils.get_tagged_from_server("who played cruella deville in 102 dalmatians?", caseless=True), ngram_len=2)]
    [['played', 'cruella'], ['cruella', 'deville'], ['102', 'dalmatians']]
    >>> [f['tokens'] for f in extract_entities(utils.get_tagged_from_server("who was the winner of the 2009 nobel peace prize?", caseless=True), ngram_len=4)]
    [['2009', 'nobel', 'peace', 'prize'], ['winner', 'of', 'the', '2009']]
    >>> [f['tokens'] for f in extract_entities(utils.get_tagged_from_server("who is the senator of connecticut 2010?", caseless=True), ngram_len=1)]
    [['connecticut'], ['senator'], ['2010']]
    >>> [f['tokens'] for f in extract_entities(utils.get_tagged_from_server("who is the senator of connecticut 2010?", caseless=True), ngram_len=1)]
    [['connecticut'], ['senator'], ['2010']]
    >>> [f['tokens'] for f in  extract_entities(utils.get_tagged_from_server('I be doin me, dgaf bout what you doin'), ngram_len=3)]
    [['I', 'be', 'doin'], ['me', ',', 'dgaf'], ['dgaf', 'bout', 'what'], ['bout', 'what', 'you'], ['what', 'you', 'doin'], ['what', 'you']]
    """
    for i, t in enumerate(tagged_input):
        t['abs_id'] = i
    chunks = np_parser.parse(tagged_input, ngram_len)
    chunks = [n for n in chunks if len({el['abs_id'] for el in n} & linked_token_ids) == 0]
    fragments = []
    for n in chunks:
        fragment_type = ("NNP" if any(el['pos'] in {'NNP', 'NNPS'} for el in n) else
                         "NN" if any(el['pos'] != "CD" and el['ner'] != "DATE" for el in n) else
                         "DATE" if all(el['ner'] == "DATE" for el in n) else utils.unknown_el)
        fragment = {
            'type': fragment_type,
            'tokens': [el['word'] for el in n],
            'token_ids': [el['abs_id'] for el in n],
            'poss': [el['pos'] for el in n],
            'offsets': (min(el['characterOffsetBegin'] for el in n), max(el['characterOffsetEnd'] for el in n))
        }
        fragments.append(fragment)
    fragments = [f for f in fragments if f['type'] == "NNP"] + \
                [f for f in fragments if f['type'] == "NN"] + \
                [f for f in fragments if f['type'] == "DATE"]
    return fragments


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
