"""
Optimised MedScribe — Local Agreement Buffer

This module solves the DUPLICATE WORDS problem in live transcription.

THE PROBLEM:
    In live transcription, we process OVERLAPPING audio windows.  We
    intentionally overlap windows so Whisper has context and doesn't
    miss words at the boundary between chunks.

    Example with 2s windows and 0.75s overlap:
        Window 1: [0.00s ——————————————— 2.00s]
        Window 2:          [1.25s ——————————————— 3.25s]
        Window 3:                    [2.50s ——————————————— 4.50s]
                           ^overlap^

    The problem is that Whisper transcribes the overlapping portion TWICE:

        Window 1 output: "Mera naam Rahul hai"
        Window 2 output: "Rahul hai aur mujhe bukhar hai"
                          ^^^^^^^^^^
                          DUPLICATED from window 1!

    If we just concatenate these outputs, we get:
        "Mera naam Rahul hai Rahul hai aur mujhe bukhar hai"
                             ^^^^^^^^^^
                             WRONG — duplicated!

THE SOLUTION — LOCAL AGREEMENT:
    The Local Agreement algorithm compares consecutive transcriptions
    at the WORD level.  It finds where the old output ENDS and the new
    output BEGINS, identifies the overlapping words, and only emits
    the NEW words that weren't already emitted.

    Step 1: Split both transcriptions into word lists.
        old_words = ["Mera", "naam", "Rahul", "hai"]
        new_words = ["Rahul", "hai", "aur", "mujhe", "bukhar", "hai"]

    Step 2: Find the longest matching suffix of old_words at the
            start of new_words.
        old suffix: ["Rahul", "hai"]
        new prefix: ["Rahul", "hai"]  ← MATCH!  (agreement point)

    Step 3: Emit only the words AFTER the agreement point.
        new_only = ["aur", "mujhe", "bukhar", "hai"]

    Final output: "Mera naam Rahul hai aur mujhe bukhar hai"
                   ← from window 1 → ← new from window 2 →
                                       CORRECT — no duplicates!

WHY "LOCAL" AGREEMENT?
    We only compare the END of the previous output with the START of
    the new output (the "local" boundary region).  We don't need to
    compare entire transcripts, which would be slow and unnecessary.

COMPARISON WITH OLD APPROACH:
    Old (live_scribe.py):
        - Checked if text == last_text (exact full-sentence match)
        - Missed partial overlaps like "Rahul hai" appearing in both
        - Result: frequent duplicate phrases

    New (agreement.py):
        - Word-level suffix/prefix matching
        - Catches partial overlaps of any length
        - Result: clean, duplicate-free text
"""


class LocalAgreementBuffer:
    """Buffer that de-duplicates overlapping live transcription outputs.

    This buffer stores the previous transcription output and compares
    it with each new output to find and remove duplicate words.

    Usage in the live pipeline:
        buffer = LocalAgreementBuffer()

        # Window 1
        text1 = transcriber.transcribe_chunk(audio_window_1)
        new_text = buffer.process(text1)
        print(new_text)  # "Mera naam Rahul hai"

        # Window 2 (overlapping)
        text2 = transcriber.transcribe_chunk(audio_window_2)
        new_text = buffer.process(text2)
        print(new_text)  # "aur mujhe bukhar hai"  (no duplicates!)

    Attributes:
        _previous_words: The word list from the last transcription output.
        _max_overlap_words: Maximum number of words to check for overlap.
    """

    def __init__(self, max_overlap_words: int = 7):
        """Initialize the agreement buffer.

        Args:
            max_overlap_words: Maximum number of words at the end of the
                               previous output to compare against the start
                               of the new output.

                               Why limit it?
                                   With a 0.75s overlap and normal speaking
                                   speed (~150 words/min), the overlap region
                                   contains at most ~2-3 words.  Setting the
                                   max to 7 gives us a generous safety margin
                                   while keeping comparisons fast.

                                   Setting it too high would waste CPU cycles
                                   comparing words that can't possibly overlap.
        """
        self._previous_words: list[str] = []
        self._max_overlap_words = max_overlap_words

    def process(self, new_text: str) -> str:
        """Process a new transcription output and return only the new words.

        This is the main method.  It:
            1. Splits the new text into words.
            2. Finds the agreement point with the previous output.
            3. Returns only the new (non-duplicate) words.
            4. Updates the internal state for the next call.

        Args:
            new_text: The full transcription output from Whisper for
                      the current audio window.

        Returns:
            A string containing only the NEW words that should be
            emitted.  Returns empty string if everything was a duplicate.
        """
        if not new_text or not new_text.strip():
            return ""

        new_words = new_text.strip().split()

        if not self._previous_words:
            # First window — nothing to compare against, emit everything
            self._previous_words = new_words
            return new_text.strip()

        # Find the agreement point
        overlap_length = self._find_overlap(self._previous_words, new_words)

        # Extract only the new (non-duplicate) words
        if overlap_length > 0:
            new_only = new_words[overlap_length : ]
        else:
            # No overlap found — emit everything
            # This can happen if the audio windows are far apart
            # or if the transcription changed dramatically
            new_only = new_words

        # Update state for next call
        self._previous_words = new_words

        if new_only:
            return " ".join(new_only)
        else:
            # Everything in the new output was a duplicate
            return ""

    def _find_overlap(
        self, old_words: list[str], new_words: list[str]
    ) -> int:
        """Find the longest matching overlap between old suffix and new prefix.

        We look for the longest sequence of words at the END of old_words
        that matches the START of new_words.

        Algorithm:
            We try progressively shorter suffixes of old_words and check
            if they match the start of new_words.  We start with the
            longest possible overlap and return the first match.

        Example:
            old_words = ["Mera", "naam", "Rahul", "hai"]
            new_words = ["Rahul", "hai", "aur", "mujhe", "bukhar", "hai"]

            Try suffix length 4: ["Mera", "naam", "Rahul", "hai"]
                vs new prefix:   ["Rahul", "hai", "aur", "mujhe"]
                → NO MATCH

            Try suffix length 3: ["naam", "Rahul", "hai"]
                vs new prefix:   ["Rahul", "hai", "aur"]
                → NO MATCH

            Try suffix length 2: ["Rahul", "hai"]
                vs new prefix:   ["Rahul", "hai"]
                → MATCH!  Return 2.

        Args:
            old_words: Word list from the previous transcription.
            new_words: Word list from the current transcription.

        Returns:
            The number of words at the start of new_words that overlap
            with the end of old_words.  Returns 0 if no overlap found.
        """
        # Limit the search to avoid wasting time on impossibly long overlaps
        max_check = min(
            len(old_words),
            len(new_words),
            self._max_overlap_words,
        )

        # Try from longest possible overlap down to 1 word
        for length in range(max_check, 0, -1):
            # Get the last 'length' words of old
            old_suffix = old_words[-length:]
            # Get the first 'length' words of new
            new_prefix = new_words[:length]

            if self._words_match(old_suffix, new_prefix):
                return length

        return 0

    @staticmethod
    def _words_match(words_a: list[str], words_b: list[str]) -> bool:
        """Check if two word lists match (case-insensitive).

        We use case-insensitive comparison because Whisper sometimes
        capitalizes words inconsistently between runs.  For example:
            Window 1: "rahul hai"
            Window 2: "Rahul hai aur..."

        Without case-insensitive matching, we'd miss this overlap.

        Args:
            words_a: First word list.
            words_b: Second word list (same length).

        Returns:
            True if all words match (case-insensitive).
        """
        if len(words_a) != len(words_b):
            return False

        return all(
            a.lower() == b.lower()
            for a, b in zip(words_a, words_b)
        )

    def reset(self):
        """Reset the buffer state.

        Call this when starting a new recording session.
        """
        self._previous_words = []

    @property
    def previous_text(self) -> str:
        """Get the full text of the previous transcription (for debugging).

        Returns:
            The previous transcription as a single string, or empty string.
        """
        return " ".join(self._previous_words) if self._previous_words else ""
