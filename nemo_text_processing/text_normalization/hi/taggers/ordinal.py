# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, NEMO_DIGIT, NEMO_HI_DIGIT
from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

# Convert Arabic digits (0-9) to Hindi digits (०-९)
arabic_to_hindi_digit = pynini.string_map([
    ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
    ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
]).optimize()
arabic_to_hindi_number = pynini.closure(arabic_to_hindi_digit).optimize()


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Hindi ordinals, e.g.
        १०वां -> ordinal { integer: "दसवां" }
        २१वीं -> ordinal { integer: "इक्कीसवीं" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        suffixes_list = pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv"))
        suffixes_map = pynini.string_file(get_abs_path("data/ordinal/suffixes_map.tsv"))
        # Union all suffixes for matching (includes "वाँ" which we just added)
        # Include both direct suffixes and input side of mappings for matching
        all_suffixes = pynini.union(suffixes_list, pynini.project(suffixes_map, "input"))
        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))
        
        # Load ordinal conversion data files (cardinal -> ordinal)
        # These convert cardinal words like "एक", "दो" to ordinal words like "पहला", "दूसरा"
        try:
            ordinal_digit = pynini.string_file(get_abs_path("data/ordinal/digit.tsv"))
            ordinal_teen = pynini.string_file(get_abs_path("data/ordinal/teen.tsv"))
            # Combine both mappings
            cardinal_to_ordinal = pynini.union(ordinal_digit, ordinal_teen).optimize()
        except:
            # If files don't exist, use identity (no conversion)
            cardinal_to_ordinal = pynini.closure(pynini.cross("", ""))

        # Build graph similar to decimal tagger: separate paths for Hindi and Arabic digits
        # Pattern: Hindi or Arabic digits followed by ordinal suffixes
        cardinal_graph = cardinal.final_graph
        
        # Hindi digits path: Hindi digits + suffix -> delete suffix -> convert to words -> add suffix
        hindi_digits = pynini.closure(NEMO_HI_DIGIT, 1)
        hindi_ordinal_pattern = hindi_digits + pynutil.delete(all_suffixes)
        # Convert Hindi digits -> cardinal words (like decimal tagger does)
        hindi_cardinal = pynini.compose(hindi_ordinal_pattern, cardinal_graph).optimize()
        
        # Arabic digits path: Arabic digits + suffix -> delete suffix -> convert to Hindi -> convert to words
        arabic_digits = pynini.closure(NEMO_DIGIT, 1)
        arabic_ordinal_pattern = arabic_digits + pynutil.delete(all_suffixes)
        # Convert Arabic digits to Hindi digits, then to cardinal words (like decimal tagger does)
        arabic_cardinal = pynini.compose(
            arabic_ordinal_pattern,
            arabic_to_hindi_number @ cardinal_graph
        ).optimize()
        
        # Combine both paths
        combined_cardinal = hindi_cardinal | arabic_cardinal
        
        # Apply cardinal to ordinal conversion
        # Try mapping first (for 1-4: converts to special forms like पहला, दूसरा)
        # If no match, use cardinal as-is
        ordinal_base = pynini.union(
            pynini.compose(combined_cardinal, cardinal_to_ordinal),
            combined_cardinal
        ).optimize()
        
        # Add suffix back - append suffix to the end
        # For multi-word outputs like "एक सौ", append to last word
        graph = ordinal_base + suffixes_list
        
        exceptions = pynutil.add_weight(exceptions, -0.1)
        graph = pynini.union(exceptions, graph)

        # Store graph before tokenization (needed for serial tagger)
        # This graph has Hindi digits + suffixes on input, Hindi words + suffix on output
        self.graph = graph.optimize()

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
