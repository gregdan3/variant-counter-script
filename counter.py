import csv
import hashlib
import itertools
import json
import re
import math
import uuid
from collections import defaultdict, OrderedDict
from collections.abc import Generator, Iterable
from datetime import datetime
from typing import Any, TypedDict, TypeVar, cast
from uuid import UUID

# from scipy.stats import norm

from sonatoki.Configs import CorpusConfig
from sonatoki.ilo import Ilo
from copy import deepcopy

CONFIG = deepcopy(CorpusConfig)
CONFIG["empty_passes"] = False
ILO = Ilo(**CONFIG)


SPDATA_FILE = "./sp-data.csv"

FROM_PLATFORM = "tan (yes)"
NOT_FROM_PLATFORM = "ala (no)"

RAW_FIELDS = [
    "submitted_at",
    "image_url",
    "from_platform",
    "source_url",
    "platform",
    "created_at",
    "author",
    "source_url_open",
    "medium",
    "created_at_open",
    "author_open",
    "transcript",
    "comment",
    *(["_"] * 35),
    "variants",
    "_",
    "features",
]

T = TypeVar("T")


class Sentence(TypedDict):
    words: list[str]
    author: UUID


class HitsData(TypedDict):
    hits: int
    authors: set[UUID]


class HitsProc(TypedDict):
    hits: int
    authors: int


Metacounter = dict[int, dict[str, HitsData]]
MetacounterProc = dict[int, dict[str, HitsProc]]


def overlapping_ntuples(iterable: Iterable[T], n: int) -> Iterable[T]:
    teed = itertools.tee(iterable, n)
    for i in range(1, n):
        for j in range(i):
            _ = next(teed[i], None)
            # offset start by position

    # ends when any iter is empty; all groups will be same size
    return zip(*teed)


def overlapping_terms(iterable: Iterable[str], n: int) -> Iterable[str]:
    return [" ".join(item) for item in overlapping_ntuples(iterable, n)]


def string_to_uuid(input_string: str) -> uuid.UUID:
    hash_bytes = hashlib.sha1(input_string.encode("utf-8")).digest()
    return uuid.UUID(bytes=hash_bytes[:16], version=5)


def count_frequencies(
    sents: Iterable[Sentence],
    max_term_len: int,
) -> Metacounter:
    metacounter: Metacounter = {
        term_len: defaultdict(lambda: HitsData({"hits": 0, "authors": set()}))
        for term_len in range(1, max_term_len + 1)
    }
    for sent in sents:
        words = sent["words"]
        author = sent["author"]
        sent_len = len(words)
        if not sent_len:
            continue

        for term_len in range(1, max_term_len + 1):
            if sent_len < term_len:
                continue

            terms = overlapping_terms(words, term_len)
            for term in terms:
                metacounter[term_len][term]["hits"] += 1
                metacounter[term_len][term]["authors"] |= {author}

    return metacounter


def process_metacounter(metacounter: Metacounter) -> MetacounterProc:
    for term_len, terms in metacounter.items():
        for term, data in terms.items():
            data["authors"] = len(data["authors"])
        sorted_terms = OrderedDict(
            sorted(terms.items(), key=lambda item: item[1]["hits"], reverse=True)
        )
        metacounter[term_len] = sorted_terms

    return metacounter


class RawRow(TypedDict):
    submitted_at: str  # "Timestamp"
    image_url: str  # "o pana e sitelen."
    from_platform: str  # "sitelen li tan ala tan ma ilo kulupu? (ni li ken: ilo Discord, lipu Facebook, ilo Telegram, ilo YouTube). tan la sina o sona e ni: seme li pana e sitelen lon tenpo seme?"
    source_url: str  # "o pana e nasin tawa sitelen lon ma"
    platform: str
    created_at: str  # "ni li tan tenpo seme?"
    author: str  # "seme li sitelen e ni? o pana e nimi ona lon ma"

    source_url_open: str
    medium: str  # "ni li tan ma seme?" or "sitelen li tan seme li lon nasin seme?"
    created_at_open: str
    author_open: str

    transcript: str  # "o pana e sitelen Lasina pi sitelen ni. nimi li lon poki nimi la o pana e kalama nimi."
    comment: str  # "sina wile toki ante la o toki!"
    variants: str  # "variants used"
    features: str  # "other variations"
    _: str


class ProcessedRow(TypedDict):
    submitted_at: datetime
    from_platform: bool
    image_url: str
    source_url: str | None
    medium: str
    created_at: datetime | None
    author: str
    transcript: str
    comment: str
    variants: dict[str, int]
    features: dict[str, bool]


def count_variants(variants: str) -> dict[str, int]:
    variants = variants.lower()
    variants = variants.replace('"', "")
    variants = variants.replace("'", "")
    lines = variants.split("\n")
    pattern = re.compile(r"(?P<name>.+?) is used(?: (?P<count>\d+) time(?:s)?)?\.")

    result: dict[str, int] = dict()
    for line in lines:
        match = pattern.search(line)
        if not match:
            continue
        name = match.group("name")
        assert name
        count = int(match.group("count") or 1)  # Default to 1 if count is not specified
        result[name] = result.get(name, 0) + count

    return result


def note_features(features: str) -> dict[str, bool]:
    lines = features.split("\n")
    return dict()


def to_datetime(datetime_str: str) -> datetime | None:
    if not datetime_str:
        return None
    # "%m/%d/%Y %H:%M:%S"
    # unused format
    date_str = datetime_str
    if " " in date_str:
        date_str = date_str.split(" ")[0]
    return datetime.strptime(date_str, "%m/%d/%Y")


def process_row(row: RawRow) -> ProcessedRow:
    processed: ProcessedRow = dict()  # pyright: ignore[reportAssignmentType]

    processed["submitted_at"] = cast(datetime, to_datetime(row["submitted_at"]))
    processed["image_url"] = row["image_url"]
    processed["transcript"] = row["transcript"]
    processed["comment"] = row["comment"]
    processed["variants"] = count_variants(row["variants"])
    processed["features"] = note_features(row["features"])

    if row["from_platform"] == FROM_PLATFORM:
        processed["from_platform"] = True
        processed["author"] = row["author"]
        processed["source_url"] = row["source_url"]
        processed["medium"] = row["platform"]
        processed["created_at"] = cast(datetime, to_datetime(row["created_at"]))

    elif row["from_platform"] == NOT_FROM_PLATFORM:
        processed["from_platform"] = False
        processed["source_url"] = row["source_url_open"]
        processed["medium"] = row["medium"]
        processed["created_at"] = to_datetime(row["created_at_open"])
        processed["author"] = row["author_open"]

    else:
        raise Exception(f"Row platform is not marked properly: {row}")

    return processed


def fetch_rows(csv_file: str) -> Generator[ProcessedRow]:
    with open(csv_file) as f:
        reader = csv.DictReader(f, RAW_FIELDS, strict=True)
        _ = next(reader, None)  # skip line 0 (headers)
        for row in reader:
            row = cast(RawRow, row)  # pyright: ignore[reportInvalidCast]
            yield process_row(row)


def row_to_sentences(row: ProcessedRow) -> list[Sentence]:
    sentences: list[Sentence] = list()
    for scorecard in ILO.make_scorecards(row["transcript"]):
        if scorecard["score"] < 0.8:
            continue

        sentence = Sentence(
            {
                "author": string_to_uuid(row["author"]),
                "words": [word.lower() for word in scorecard["cleaned"]],
            }
        )
        sentences.append(sentence)

    return sentences


def process_variants(rows: list[ProcessedRow]) -> dict[str, int]:
    all_variants: dict[str, int] = dict()
    for row in rows:
        variants = row["variants"]
        for key, value in variants.items():
            all_variants[key] = all_variants.get(key, 0) + value

    sorted_variants = dict(
        sorted(all_variants.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_variants


def dump_to_json(obj: dict[Any, Any], filename: str):
    with open(filename, "w") as f:
        f.write(json.dumps(obj, default=str, indent=2))


def get_total_hits_authors(frequencies: Metacounter) -> tuple[int, int]:
    total_hits = 0
    total_authors_set: set[UUID] = set()
    total_authors = 0
    for length, data in frequencies.items():
        if length > 1:
            break
        for word, hitsdata in data.items():
            total_hits += hitsdata["hits"]
            total_authors_set |= hitsdata["authors"]
    total_authors = len(total_authors_set)
    return total_hits, total_authors


def norm_cdf(z: float) -> float:
    return (1 + math.erf(z / math.sqrt(2))) / 2


def find_max_confidence(instance_samples: int, total_samples: int) -> float:
    p = instance_samples / total_samples
    max_std_err = p * (1 - p)  # p and its inverse
    std_err = math.sqrt(max_std_err / total_samples)
    max_margin_err = abs(0.5 - p) * 0.999
    # make it a tiny bit smaller so we still have a majority in the worst case

    z = max_margin_err / std_err
    confidence_level = 2 * norm_cdf(z) - 1
    return confidence_level


def error_check(row: ProcessedRow, sentences: list[Sentence]):
    variants = row["variants"]
    # TODO: check that every variant appears in a given row's transcript
    # for
    # *_, word = variants.split(" ")


def print_result(identifier: str, hits: int, total: int):
    p = hits / total
    if p == 0.5:
        print(f"{identifier}: {hits}/{total} ({p * 100:.2f}%)")
        print(f"{identifier} is tied (0.00% confidence)\n")

    confidence = find_max_confidence(hits, total)
    is_majority = "not a majority"
    if p > 0.5:
        is_majority = "a majority"

    print(f"{identifier}: {hits}/{total} ({p * 100:.2f}%)")
    print(f"{confidence* 100:.2f}% confidence that {identifier} is {is_majority}\n")


def main():
    all_rows: list[ProcessedRow] = list()
    all_sentences: list[Sentence] = list()
    for row in fetch_rows(SPDATA_FILE):
        all_rows.append(row)
        sentences = row_to_sentences(row)
        all_sentences.extend(sentences)

    frequencies = count_frequencies(all_sentences, 2)
    total_hits, total_authors = get_total_hits_authors(frequencies)

    variants = process_variants(all_rows)
    long_variants = 0
    total_longable_hits = 0

    for variant, variant_hits in variants.items():
        _pieces = variant.split(" ")
        determiner = _pieces[0]
        word = _pieces[-1]

        word_hits = frequencies[1][word]["hits"]
        if word_hits <= 1:
            continue

        if determiner == "long":
            long_variants += variant_hits
            total_longable_hits += word_hits

        print_result(variant, variant_hits, word_hits)

    print_result("Glyph extensions", long_variants, total_longable_hits)

    frequencies = process_metacounter(frequencies)
    dump_to_json(frequencies, "freqs.json")
    dump_to_json(variants, "variants.json")


if __name__ == "__main__":
    main()
