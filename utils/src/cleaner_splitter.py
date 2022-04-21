import argparse
import sys
import unicodedata

from .sentence_split import get_split_algo
from .remove_non_printing_char import get_replacer as non_printing_char_replacer

import xxhash
from sacremoses import MosesPunctNormalizer


class SentenceSplitClean:
    def __init__(self, splitter_lang: str, split_algo: str):
        # setup sentence splitter
        self.splitter = get_split_algo(splitter_lang, split_algo=split_algo)

        # setup "moses" normalization
        self.mpn = MosesPunctNormalizer(lang="en")  # TODO
        self.replace_nonprint = non_printing_char_replacer(" ")

    def __call__(self, line):
        sentence_splits = self.splitter(line)
        line_hash = xxhash.xxh3_64_intdigest(line)

        for sent in sentence_splits:
            # normalize -- moses equivalent
            clean = self.mpn.normalize(sent)
            clean = self.replace_nonprint(clean)
            # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
            clean = unicodedata.normalize("NFKC", clean)

            yield (line_hash, sent, clean)


def split_clean(splitter_lang: str, split_algo: str):
    sentence_split_clean = SentenceSplitClean(splitter_lang, split_algo)

    for line in sys.stdin:
        line = line.strip()

        for line_hash, sent, clean in sentence_split_clean(line):
            print(f"{line_hash}\t{sent}\t{clean}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--algo", default="default")
    args = parser.parse_args()

    split_clean(args.lang, args.algo)


if __name__ == "__main__":
    main()
