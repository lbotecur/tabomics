import re

import tabomics


def test_version_is_a_pep440_string():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+|rc\d+)?", tabomics.__version__)
