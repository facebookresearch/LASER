# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Remove non printable char as per:
#  https://stackoverflow.com/questions/92438/stripping-non-printable-characters-from-a-string-in-python
#
# This is supposed to be a drop in replacement to moses strip-non-printing-char.perl

import sys
import unicodedata


def get_replacer(replace_by: str = " ") -> str:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


def test_remove():
    replaceby_ = get_replacer("_")

    assert (
        replaceby_("See what's hidden in your string…	or be​hind﻿")
        == "See what's hidden in your string…_or be_hind_"
    )

    replacebyspace = get_replacer(" ")

    assert replacebyspace("\x00\x11Hello\u200bWorld") == "  Hello World"
