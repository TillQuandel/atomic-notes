"""Generate examples/zettelkasten-primer.pdf from the embedded primer text.

This script requires fpdf2, which is NOT a project dependency.
Install it before running:

    pip install fpdf2

Then run from the repo root:

    python examples/generate_example_pdf.py

The generated PDF is committed to the repository so end users do not need
to regenerate it.
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

TITLE = "A Short Primer on Atomic Note-Taking"

SECTIONS = [
    (
        "What Is an Atomic Note?",
        [
            (
                "An atomic note captures exactly one idea. The word 'atomic' is borrowed from "
                "chemistry: just as an atom is the smallest unit that still has the properties "
                "of an element, an atomic note is the smallest unit of thought that can stand "
                "alone and still be meaningful. It has a clear title, a self-contained body, "
                "and needs no surrounding context to be understood."
            ),
            (
                "The constraint of one idea per note is not a limitation -- it is a design "
                "decision. When every note carries a single coherent claim, retrieval becomes "
                "precise and recombination becomes possible. A note that mixes three ideas is "
                "hard to link to anything because it is always half-relevant. A note about one "
                "idea can be linked wherever that idea matters, which turns the collection into "
                "a navigable network rather than a pile of files."
            ),
        ],
    ),
    (
        "The Zettelkasten Method",
        [
            (
                "The Zettelkasten -- German for 'slip box' -- is a knowledge management method "
                "developed to support long-term intellectual work. Its central insight is that "
                "knowledge does not accumulate by storing raw material; it accumulates by "
                "transforming raw material into discrete, interlinked ideas. Each slip (Zettel) "
                "receives a unique identifier, a self-contained formulation of one thought, and "
                "explicit references to related slips."
            ),
            (
                "The identifier is not merely an address. In the original physical system, a "
                "slip could be filed behind another slip by adding a letter to its number "
                "(1a, 1a1, 1b, ...), which allowed branching trains of thought without "
                "disrupting the existing sequence. Digital implementations replace this physical "
                "addressing with hyperlinks or wikilinks, but the underlying principle is the "
                "same: connections between notes are first-class citizens, not an afterthought."
            ),
        ],
    ),
    (
        "Progressive Summarization",
        [
            (
                "Progressive summarization is a reading and note-taking strategy that processes "
                "source material in layers. In the first pass, you highlight the passages that "
                "seem most relevant. In the second pass, you bold the highlights that are most "
                "essential. Later, you distill the bolded text into a short summary in your own "
                "words at the top of the note. Each layer is more compressed and more personal "
                "than the layer below it."
            ),
            (
                "The key discipline is restraint in the early passes. If you bold everything, "
                "you have summarized nothing. The value of progressive summarization emerges "
                "from the selectivity: you are constantly asking 'what is this really about?' "
                "rather than 'what do I want to remember?'. The first question produces durable "
                "distillations; the second tends to produce notes that make sense the day they "
                "are written and are confusing six months later."
            ),
        ],
    ),
    (
        "Linking and Emergence",
        [
            (
                "Linking is the act of creating a connection between two notes that share a "
                "conceptual relationship. In a digital system, this is a hyperlink or a "
                "wikilink. In a physical Zettelkasten, it is a written reference: 'see also: "
                "slip 42b'. The mechanics are trivial; the discipline is not. A link is a "
                "claim: these two ideas are related in a way worth encoding. Casual linking -- "
                "connecting everything to everything -- produces a hairball. Deliberate linking "
                "produces structure."
            ),
            (
                "Emergence is what happens after enough deliberate linking. Patterns appear "
                "that were not visible when the individual notes were written. A cluster of "
                "notes that all link to the same hub note suggests a concept worth unpacking. "
                "A chain of notes that each reference the next reveals a logical progression. "
                "A note with no incoming links is either an orphan that needs integration or a "
                "genuine outlier worth inspecting. The structure of the network is a form of "
                "knowledge that cannot be read from any individual note -- it only exists in "
                "the relations."
            ),
            (
                "The practical implication is that building a Zettelkasten is not primarily a "
                "writing task; it is a thinking task. You are not filing information for later "
                "retrieval. You are externalizing a reasoning process so that the system can "
                "reflect structure back to you. The atomic note is the unit. The link is the "
                "operation. Emergence is the result."
            ),
        ],
    ),
]


def build_pdf(output_path: str) -> None:
    pdf = FPDF()
    pdf.compress = False  # keep file readable / larger for test assertions
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", style="B", size=18)
    pdf.cell(0, 12, TITLE, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(6)

    for section_title, paragraphs in SECTIONS:
        # Section heading
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 9, section_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        pdf.set_font("Helvetica", size=11)
        for para in paragraphs:
            pdf.multi_cell(0, 6, para)
            pdf.ln(4)

        pdf.ln(2)

    pdf.output(output_path)
    print(f"Written: {output_path}")


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(script_dir, "zettelkasten-primer.pdf")
    build_pdf(output)
