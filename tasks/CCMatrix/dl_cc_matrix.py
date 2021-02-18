import contextlib
import gzip
import logging
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Type

from cc_net.jsonql import open_remote_file, open_write
from cc_net.process_wet_file import CCSegmentsReader
from typing import Sequence
import functools
import multiprocessing

BUFFER_SIZE = "32G"
SORT_PARALLEL = 8

KNOWN_VERSIONS = ["v1.0.0", "v1.0.beta", "v1.0.alpha"]


class NormalizedBitextPtr(NamedTuple):
    lang_pair: str
    line_no: int
    segment: str
    digest: str
    ptr_start: int
    ptr_end: int
    score: float


class Bitext(NamedTuple):
    lang_pair: str
    line_no: int
    score: float
    text: str


class SimpleBitext(NamedTuple):
    line_no: int
    score: float
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


def dl(outdir: Path = Path("data"), version: str = KNOWN_VERSIONS[0], parallelism: int = 8):
    """
    Download bitext pointers from FAIR dataset and extract corresponding CC snippets.
    - version: Specific version to download
    - outdir: Directory where the data should go. Files will be in {outdir}/{version}/raw/
    """
    assert version in KNOWN_VERSIONS, f"Unknown version {version}, chose from {KNOWN_VERSIONS}"
    metadata_dir = f"https://dl.fbaipublicfiles.com/laser/CCMatrix/{version}"
    file_list = [l.strip() for l in open_remote_file(metadata_dir + "/list.txt")]
    outdir.mkdir(exist_ok=True)
    outdir = outdir / version / "raw"
    outdir.mkdir(exist_ok=True, parents=True)

    dlf = functools.partial(dl_file, metadata_dir, outdir)
    # list(map(dlf, file_list))
    with multiprocessing.Pool(parallelism) as pool:
        pool.map(dlf, file_list)


def get_documents(segment: str) -> Dict[str, str]:
    return {d["digest"]: d["raw_content"] for d in CCSegmentsReader([segment])}


def dl_file(metadata_dir: str, outdir: Path, file: str):
    metadata = "/".join((metadata_dir, file))
    parser = get_typed_parser(NormalizedBitextPtr)
    found_bitext, missed_bitext, skipped_line = 0, 0, 0
    segment = ""
    segment_downloads: Dict[str, int] = defaultdict(int)
    raw_documents: Dict[str, str] = {}
    cleaned_documents: Dict[str, str] = {}

    outfile = outdir / file
    if outfile.exists():
        return
    o = FileWriterWithTmp(outfile)
    for i, line in enumerate(open_remote_file(metadata)):
        try:
            bitext: NormalizedBitextPtr = parser(line)
            # Add some more assert in case the line is invalid but still parse
            assert bitext.segment.startswith("crawl-data/")
            assert bitext.digest.startswith("sha1:")
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
        score = getattr(bitext, "score", 0.0)
        bt = Bitext(bitext.lang_pair, bitext.line_no, score, text)
        print(*bt, sep="\t", file=o)

    o.close(True)
    logging.info(f"Found {found_bitext} sentences, missed {missed_bitext} sentences.")
    if skipped_line > 0:
        logging.error(f"Skipped {skipped_line} unparsable lines")
    expected_dl = len(segment_downloads)
    actual_dl = sum(segment_downloads.values())

    if actual_dl != expected_dl:
        logging.error(
            f"Some segments where downloaded twice. Total dl: {actual_dl}, distinct dl: {expected_dl}"
        )


def _tmp(file: Path) -> Path:
    tmp_dir = file.parent
    prefix = file.name.split(".", 1)[0] + "."
    suffix = ".tmp." + file.name[len(prefix) :]
    _, tmp_path = tempfile.mkstemp(dir=tmp_dir, prefix=prefix, suffix=suffix)
    return Path(tmp_path)


class FileWriterWithTmp:
    def __init__(self, file: Path):
        self.file = file
        self.tmp_file = _tmp(file)
        # We don't want to make FileWriterWithTmp a ContextManager
        self.handle = open_write(self.tmp_file).__enter__()

    def write(self, data) -> int:
        return self.handle.write(data)

    def close(self, success: bool = False):
        self.handle.close()
        if success:
            self.tmp_file.rename(self.file)


def transpose_file(outdir: Path, file: Path) -> None:
    sentinel_file = file.with_suffix(".transposed")
    if sentinel_file.exists():
        return
    outputs: Dict[str, FileWriterWithTmp] = {}
    parser = get_typed_parser(Bitext)
    success = False
    try:
        for line in open_read(file):
            bt: Bitext = parser(line)
            lang_pair = bt.lang_pair
            if bt.lang_pair not in outputs:
                assert (
                    "/" in lang_pair
                ), f"Invalid lang pair '{lang_pair}' should be 'src-trg/src' or 'src-trg/trg'"
                (outdir / f"{lang_pair}").mkdir(exist_ok=True, parents=True)
                o = FileWriterWithTmp(outdir / f"{lang_pair}_{file.name}")
                outputs[lang_pair] = o
            simple_bt = SimpleBitext(bt.line_no, bt.score, bt.text)
            print(*simple_bt, sep="\t", file=outputs[lang_pair])
        success = True
    finally:
        for o in outputs.values():
            o.close(success)
        if success:
            sentinel_file.write_text("\n".join(str(o.file) for o in outputs.values()))
            # file.unlink()


def sort_files(outdir: Path, lang_pair_dir: Path, lang: str) -> Path:
    out = outdir / lang_pair_dir.name / f"{lang}.txt"
    if out.exists():
        return out

    files: List[Path] = []
    for f in lang_pair_dir.iterdir():
        if not f.suffix == ".gz":
            continue
        if f.name.split("_")[0] != lang:
            continue
        files.append(f)

    print(f"Found {len(files)} files for lang '{lang}' in {lang_pair_dir}: {files}")
    assert len(files) > 0

    (outdir / lang_pair_dir.name).mkdir(exist_ok=True, parents=True)
    tmp_out = _tmp(out)
    
    unzipped_files = []
    for f in files:
        subprocess.check_call(["gunzip", "-k", str(f)])
        unzipped_files.append(str(f)[:-3])

    sort_cmd = [
        "sort",
        "-nk1",
        f"--parallel={SORT_PARALLEL}",
        f"--buffer-size={BUFFER_SIZE}",
        "--output",
        str(tmp_out),
        ] + unzipped_files
    subprocess.check_call(sort_cmd)
    tmp_out.rename(out)
    return out


def finalize(
    outdir: Path = Path("data"), version: str = KNOWN_VERSIONS[0], pairs: Sequence[str] = []
) -> None:
    """From the downloaded raw text files, extract the bitexts, sorted by language pair.
    Assumes 'dl' has been run with the same outdir and version before.

    - version: Specific version to download
    - outdir: Directory where the data should go. Files will be in {outdir}/{version}/bitext/
    - pairs: List of language pairs you are interested in. Defaults to all.
    """
    raw_dir = outdir / version / "raw"
    if not raw_dir.is_dir():
        cmd = f"python {__file__} dl --outdir {outdir} --version {version}"
        assert raw_dir.is_dir(), f"Dir not found {raw_dir}. Did you run following command?\n{cmd}"

    raw_files = list(raw_dir.glob("*.gz"))
    split_dir = outdir / version / "split_by_lang"
    split_dir.mkdir(exist_ok=True, parents=True)
    tr = functools.partial(transpose_file, split_dir)
    with multiprocessing.Pool() as pool:
        pool.map(tr, raw_files)

    bitext_dir = outdir / version / "bitext"
    bitext_dir.mkdir(exist_ok=True, parents=True)
    if pairs:
        pair_dirs = []
        for pair in pairs:
            assert (
                len(pair.split("-")) == 2
            ), f"Invalid pair '{pair}', should be 'src-trg'"
            pair_dir = split_dir / pair
            assert (
                pair_dir.is_dir()
            ), f"Dir {pair_dir} not found for lang pair '{pair}'. Is the pair valid ?"
            pair_dirs.append(pair_dir)
    else:
        pair_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

    for pair_dir in pair_dirs:
        src, trg = pair_dir.name.split("-")
        src_file = sort_files(bitext_dir, pair_dir, src)
        trg_file = sort_files(bitext_dir, pair_dir, trg)
        validate(src_file, trg_file)


def validate(src_file: Path, trg_file: Path) -> None:
    """Checks that the segments in the given batch are valid."""
    lines_src, lines_trg, found_pairs = 0, 0, 0
    parser = get_typed_parser(SimpleBitext)
    with open(src_file) as src_f, open(trg_file) as trg_f:
        src_l = src_f.readline()
        trg_l = trg_f.readline()
        while src_l and trg_l:
            src: SimpleBitext = parser(src_l)
            trg: SimpleBitext = parser(trg_l)
            if src.line_no <= trg.line_no:
                lines_src += 1
                src_l = src_f.readline()
            if trg.line_no <= src.line_no:
                lines_trg += 1
                trg_l = trg_f.readline()
            if trg.line_no == src.line_no:
                found_pairs += 1

    if found_pairs == lines_src and found_pairs == lines_trg:
        logging.info(
            f"Validated {src_file} and {trg_file}. Found {found_pairs} bitexts."
        )
    else:
        logging.error(
            f"Validated {src_file} and {trg_file}. "
            f"Found {found_pairs} bitexts, from {lines_src} in {src_file} and {lines_trg} in {trg_file}"
        )


if __name__ == "__main__":
    import func_argparse

    func_argparse.main(dl, finalize)
