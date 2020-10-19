import gzip
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Type

from cc_net.jsonql import open_remote_file
from cc_net.process_wet_file import CCSegmentsReader


class NormalizedBitextPtr(NamedTuple):
    lang_pair: str
    line_no: int
    segment: str
    digest: str
    ptr_start: int
    ptr_end: int


class Bitext(NamedTuple):
    lang_pair: str
    line_no: int
    text: str


WEB_PAT = re.compile(r"https?:[^ \n]* ")
WEB_REPL = "WEB "

WEB2_PAT = re.compile(r"https?:[^ \n]*\n")
WEB2_REPL = "WEB\n"


def clean_content(raw_content: str) -> str:
    # We need to clean all the content, because otherwise there is no way for
    # the user to know if we need to clean it or not.
    par = raw_content
    par = par.replace("</s>", ". ")
    par = par.replace("\t", " ")
    par = re.sub(WEB_PAT, WEB_REPL, par, count=0)
    par = re.sub(WEB2_PAT, WEB2_REPL, par, count=0)
    return par


def get_typed_parser(cls: Type) -> Callable:
    types = cls.__annotations__.values()

    def parser(line: str) -> NamedTuple:
        parts = line.rstrip("\n").split("\t")
        assert len(parts) == len(
            types
        ), f"Print size mismatch expected the following columns {cls.__annotations__} got: {parts}"
        return cls(*(t(p) for t, p in zip(types, parts)))

    return parser


def open_read(file: Path) -> Iterable[str]:
    if file.suffix == ".gz":
        reader = gzip.open(file, "rt")
    else:
        reader = open(file, "rt")
    with reader as f:
        for line in f:
            yield line


def dl(outdir: Path = Path("data"), version: str = "v1.0"):
    """Checks that the segments in the given batch are valid."""
    metadata_dir = f"https://dl.fbaipublicfiles.com/laser/CCMatrix/{version}"
    file_list = [l.strip() for l in open_remote_file(metadata_dir + "/list.txt")]
    outdir.mkdir(exist_ok=True)
    outdir = outdir / version
    outdir.mkdir(exist_ok=True)

    for file in file_list:
        dl_file(metadata_dir, file, outdir)


def get_documents(segment: str) -> Dict[str, str]:
    return {d["digest"]: d["raw_content"] for d in CCSegmentsReader([segment])}


def dl_file(metadata_dir: str, file: str, outdir: Path):
    metadata = "/".join((metadata_dir, file))
    parser = get_typed_parser(NormalizedBitextPtr)
    found_bitext, missed_bitext, skipped_line = 0, 0, 0
    segment = ""
    segment_downloads: Dict[str, int] = defaultdict(int)
    raw_documents: Dict[str, str] = {}
    cleaned_documents: Dict[str, str] = {}

    outfile = outdir / file
    with gzip.open(outfile, "wt") as o:
        for i, line in enumerate(open_remote_file(metadata)):
            try:
                bitext: NormalizedBitextPtr = parser(line)
            except AssertionError:
                logging.error(f"Skipping line {i}: {line}")
                skipped_line += 1
                continue

            if not segment or bitext.segment != segment:
                segment = bitext.segment
                segment_downloads[segment] += 1
                # Load segment in RAM, purge document cache
                raw_documents = get_documents(segment)
                cleaned_documents = {}

            raw_doc = raw_documents.get(bitext.digest)
            if raw_doc is None:
                logging.error(f"Document not found: {bitext.digest} in {segment}")
                missed_bitext += 1
                continue

            clean_doc = cleaned_documents.get(bitext.digest)
            if clean_doc is None:
                clean_doc = clean_content(raw_doc)
                cleaned_documents[bitext.digest] = clean_doc

            text = clean_doc[bitext.ptr_start : bitext.ptr_end]
            bt = Bitext(bitext.lang_pair, bitext.line_no, text)
            print(*bt, sep="\t", file=o)

    logging.info(f"Found {found_bitext} sentences, missed {missed_bitext} sentences.")
    if skipped_line > 0:
        logging.error(f"Skipped {skipped_line} unparsable lines")
    expected_dl = len(segment_downloads)
    actual_dl = sum(segment_downloads.values())

    if actual_dl != expected_dl:
        logging.error(
            f"Some segments where downloaded twice. Total dl: {actual_dl}, distinct dl: {expected_dl}"
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "-o", "--outdir", type=Path, default=Path("data"), help="Target directory"
    )
    p.add_argument("-v", "--version", type=str, default="v1.0", help="Dataset version")

    args = p.parse_args()
    dl(**vars(args))
