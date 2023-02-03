import argparse
import sys
import typing as tp
import unicodedata

import xxhash
from sacremoses import MosesPunctNormalizer

from .demojizer import Demojizer, legacy_demojizer
from .remove_non_printing_char import \
    get_replacer as non_printing_char_replacer
from .sentence_split import get_split_algo

demojizer = Demojizer()


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


def remove_on_unicode_category(x: str) -> str:
    return "".join(filter(lambda ch: not unicodedata.category(ch) in {"So"}, x))


def get_replacer_unicode_category(
    skip_min: int, max_num: int, replace_by: str = " "
) -> str:
    def replace_by_unicode_category(x: str) -> str:
        total_counter = 0
        skip_counter = 0

        def flt(ch):
            nonlocal total_counter
            nonlocal skip_counter
            if max_num == 0 or total_counter < max_num:
                if unicodedata.category(ch) in {"So"}:
                    if skip_counter < skip_min:
                        skip_counter += 1
                        return ch
                    total_counter += 1
                    return replace_by
            return ch

        return "".join(map(flt, x))

    return replace_by_unicode_category


# to map with previous versions of the pipeline
def get_sentence_candidate_modifiers() -> tp.List[tp.Callable]:
    return [
        lambda x: x,
        lambda x: x + " ",
        lambda x: " " + x,
        lambda x: " " + x + " ",
        lambda x: "  " + x,
        lambda x: x.rstrip(),
        lambda x: x.lstrip(),
        lambda x: " " + x.rstrip(),
        lambda x: x.strip(),
        lambda x: demojizer(x, ""),
        lambda x: demojizer(x, "").strip(),
        lambda x: " " + demojizer(x, ""),
        legacy_demojizer,
        remove_on_unicode_category,
        get_replacer_unicode_category(1, 1),
        get_replacer_unicode_category(0, 0),
    ]


def reach_sentence_from_paragraph(
    paragraph: str,
    expected_paragraph_digest: int,
    expected_sentence_digest: int,
    lang: str,
    sentence_splitters: tp.Dict[str, "SentenceSplitClean"],
    debug_candidates: bool,
):
    if lang not in sentence_splitters:
        sentence_splitters[lang] = SentenceSplitClean(lang, "default")

    def no_splitter(paragraph):
        line_h = xxhash.xxh3_64_intdigest(paragraph)
        return [(line_h, paragraph, paragraph)]

    sentence_splitter = sentence_splitters[lang]
    splitter_candidates = [sentence_splitter, no_splitter]
    for duct_candidate in get_sentence_candidate_modifiers():
        for split_cand in splitter_candidates:
            for line_hash, sent, clean in split_cand(paragraph):
                assert line_hash == expected_paragraph_digest
                clean_cand = duct_candidate(clean)
                reached_sentence_digest = xxhash.xxh3_64_intdigest(clean_cand)
                if debug_candidates:
                    print(f"{reached_sentence_digest}::\t::{clean_cand}::")
                if reached_sentence_digest == expected_sentence_digest:
                    return clean_cand

    return None


def split_clean():
    split_algo = "default"
    sentence_splitters = {}

    for line in sys.stdin:
        line_stripped = line.rstrip("\n")
        metadata, paragraph = line_stripped.split("\t")
        (
            _,
            _,
            _,
            _,
            paragraph_digest,
            sentence_digest,
            _,
            _,
            _,
            lang,
            _,
        ) = metadata.split()
        paragraph_digest = int(paragraph_digest)
        sentence_digest = int(sentence_digest)

        sentence = reach_sentence_from_paragraph(
            paragraph,
            paragraph_digest,
            sentence_digest,
            lang,
            sentence_splitters,
            False,
        )

        if sentence is not None:
            print(f"{line_stripped}\t{sentence}")
        else:
            print(
                f"Couldn't match sentence for paragraph: {paragraph_digest} sentence: {sentence_digest} lang: {lang}",
                file=sys.stderr,
            )


def main():
    split_clean()


if __name__ == "__main__":
    main()
