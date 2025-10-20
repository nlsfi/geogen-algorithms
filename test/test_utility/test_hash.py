#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from hashlib import sha256

from pandas import DataFrame

from geogenalg.utility.hash import reset_with_random_hash_index


def test_reset_random_index_generates_sha256_like_hex_strings():
    input_data = DataFrame(
        {"a": [1, 2, 3]},
        index=["a", "b", "c"],
    )

    output_data = reset_with_random_hash_index(input_data)

    sha_256_string = sha256(b"mock").hexdigest()

    assert all(
        len(index_value) == len(sha_256_string)
        for index_value in output_data.index.to_list()
    )
