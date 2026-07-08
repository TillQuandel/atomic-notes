"""Erzeugt das copyright-freie E2E-Fixture fuer tests/extractive/test_e2e.py.

Hintergrund (#155): Das urspruengliche E2E-Fixture (bates-2017.pdf) ist ein
echtes Paper und per .gitignore bewusst untracked — auf CI-Runnern fehlte es,
das pytestmark-skipif liess alle 6 E2E-Tests leer-gruen durchlaufen. Dieses
Skript erzeugt einen eigenen, frei formulierten Text (kein Zitat aus fremden
Werken; Konzept-NAMEN wie "Cognitive Load Theory" sind nicht schutzfaehig),
der committet werden darf: tests/fixtures/e2e_corpus.pdf (.gitignore-Ausnahme).

Kalibrierung auf die extractive-Pipeline (Stand #170):

- 10 Seiten, je Kapitel ein EINWORT-Header in 18pt: pdf_chunker erkennt Header
  ueber Font-Groesse (> avg * 1.15), matcht aber nur ganze Zeilen die als
  einzelnes Wort in header_words stehen — Einwort-Header brechen also pro
  Kapitel einen Chunk, Mehrwort-Header nicht.
- Jedes Kernkonzept ("Zettelkasten Method", "Cognitive Load Theory", ...) wird
  in >= 2 Kapiteln woertlich erwaehnt, damit es das Prominenz-Gate von
  plan_concepts (min_chunk_count=2) passiert.
- Konzepte sind Mehrwort-Begriffe oder kapitalisiert (ueberleben den
  #167a-Titel-Filter gegen lowercase-Einzelwoerter) und stehen in
  Definitions-Saetzen ("X is a method that ..."), auf die GLiNER bei
  threshold 0.75 zuverlaessig anspringt.
- Saetze sind >= 20 Zeichen (clean_sentence-Gate), ohne Silbentrennung und
  ohne (S. n)-Muster (Anker setzt die Pipeline selbst).

Reproduzieren: uv run python tests/fixtures_gen/make_e2e_corpus.py
(pymupdf ist Kern-Dependency, kein Extra noetig).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

# Ein Kapitel pro Seite: (Einwort-Header, Fliesstext). Eigener Text, keine Zitate.
CHAPTERS: list[tuple[str, str]] = [
    (
        "Introduction",
        "Personal knowledge management describes the everyday work of collecting, organizing, and "
        "reusing what one reads and thinks. This corpus walks through several well known ideas from "
        "that field so that an extraction pipeline has realistic material to work with. Later pages "
        "explain the Zettelkasten Method as a way of organizing notes, and they discuss Spaced "
        "Repetition as a way of keeping knowledge available over time. The corpus also covers "
        "Cognitive Load Theory, Information Foraging Theory, the Berrypicking Model, Progressive "
        "Summarization, Dual Coding Theory, and the Testing Effect. Each chapter introduces one idea "
        "in plain language and connects it to at least one other chapter. The text is written for "
        "test purposes, yet every sentence is a complete and meaningful statement. A reader should "
        "be able to open any page and find claims that can be extracted verbatim. That property "
        "matters because an extractive pipeline must never invent sentences of its own. The pages "
        "that follow therefore favor clear declarative sentences over rhetorical questions. Where a "
        "concept has a widely used name, the name is spelled out in full so that entity recognition "
        "has a fair chance. This introduction itself mentions each major concept exactly once, while "
        "the dedicated chapters repeat them several times.",
    ),
    (
        "Zettelkasten",
        "The Zettelkasten Method is a note taking method in which every idea is written on its own "
        "small note and linked to related notes. The Zettelkasten Method became widely known through "
        "the working habits of a prolific twentieth century sociologist who kept tens of thousands "
        "of linked slips. A note in a Zettelkasten carries a unique identifier, a short claim in the "
        "writer's own words, and explicit links to neighboring notes. Because links are explicit, "
        "the Zettelkasten Method turns a pile of isolated remarks into a navigable network of "
        "arguments. Writers who use the Zettelkasten Method report that drafts assemble themselves "
        "from chains of existing notes rather than from a blank page. The method rewards small, "
        "self-contained notes, which is exactly what the Atomicity Principle in the next chapter "
        "demands. Digital tools have lowered the cost of linking, but the underlying discipline has "
        "not changed since the paper era. A Zettelkasten grows slowly, and its value comes from the "
        "density of connections rather than from the raw number of notes. Critics point out that "
        "maintaining such a system takes real effort, and that a neglected Zettelkasten decays into "
        "an ordinary archive. Supporters answer that the effort is the point, because linking forces "
        "genuine engagement with the material.",
    ),
    (
        "Atomicity",
        "The Atomicity Principle states that one note should contain exactly one idea, no more and "
        "no less. When a note follows the Atomicity Principle, it can be linked, quoted, and reused "
        "without dragging unrelated content along. The principle sounds trivial, yet most beginners "
        "write notes that bundle three or four claims into a single paragraph. Splitting such "
        "bundles is the first exercise recommended to anyone adopting the Zettelkasten Method. An "
        "atomic note has a title that names its single idea, which makes search results readable at "
        "a glance. The Atomicity Principle also interacts with review workflows, because a reviewer "
        "can accept or reject one claim at a time. Progressive Summarization, described in a later "
        "chapter, works best when the underlying notes are already atomic. There is a practical "
        "test for atomicity: if you cannot state the note's idea in one sentence, the note is not "
        "atomic yet. Some writers fear that atomic notes fragment their thinking, but experience "
        "shows the opposite, since explicit links preserve context better than long documents do. "
        "The cost of atomicity is a larger number of notes, and that cost is real. The benefit is "
        "that every note remains legible years later, because it never depended on the paragraph "
        "above it.",
    ),
    (
        "Memory",
        "Spaced Repetition is a learning method that schedules reviews of a fact at growing "
        "intervals, just before the fact would otherwise be forgotten. Decades of laboratory work "
        "show that Spaced Repetition produces far more durable memory than massed practice of the "
        "same duration. Flashcard programs implement Spaced Repetition by moving well remembered "
        "cards to longer intervals and difficult cards to shorter ones. The Testing Effect is the "
        "closely related phenomenon that retrieving a memory strengthens it more than rereading the "
        "same material. Together, Spaced Repetition and the Testing Effect explain why self testing "
        "beats highlighting as a study strategy. A knowledge worker can apply both ideas by turning "
        "important notes into review questions. The schedule matters more than the total time spent, "
        "which is why cramming feels productive but fades within days. Researchers describe the "
        "underlying regularity as the spacing effect, one of the most replicated findings in the "
        "psychology of learning. Skeptics sometimes dismiss flashcards as rote learning, but "
        "retrieval practice works for conceptual material as well when the questions demand "
        "explanation rather than recognition. Later chapters connect these results to Dual Coding "
        "Theory, because imagery gives retrieval an additional pathway.",
    ),
    (
        "Cognition",
        "Cognitive Load Theory holds that working memory is sharply limited and that instruction "
        "must manage this limit deliberately. Cognitive Load Theory distinguishes intrinsic load, "
        "which comes from the difficulty of the material itself, from extraneous load, which comes "
        "from poor presentation. A cluttered slide raises extraneous load without teaching anything, "
        "which is why minimal examples outperform decorated ones. Cognitive Load Theory also "
        "describes germane load, the useful effort of building mental schemas. Designers of study "
        "material use the theory to decide what to cut, what to sequence, and what to leave for "
        "practice. Note taking systems benefit from the same analysis, because an overloaded note "
        "imposes extraneous load on the future reader. This is one more argument for the Atomicity "
        "Principle from the earlier chapter. Dual Coding Theory complements Cognitive Load Theory by "
        "showing that verbal and visual channels have partly separate capacities. When a diagram "
        "carries part of the message, the verbal channel is relieved, and total capacity grows. "
        "Worked examples are a classic application, since studying a solved problem loads memory "
        "less than solving from scratch. The practical rule is simple to state and hard to follow: "
        "remove everything that does not serve the learning goal.",
    ),
    (
        "Foraging",
        "Information Foraging Theory models a person searching for information the way ecology "
        "models an animal searching for food. Information Foraging Theory predicts that searchers "
        "follow an information scent, the cues that suggest a source will pay off. When the scent "
        "of the current source weakens, the searcher leaves it, exactly as a forager abandons a "
        "depleted berry patch. The theory turns vague talk about browsing into testable claims "
        "about time allocation between patches. Web designers apply Information Foraging Theory "
        "when they write link labels that carry strong scent, so that visitors can predict what "
        "lies behind a click. Researchers in the field measure how long people stay in a patch "
        "before moving on, and the observed patterns match the ecological models well. The "
        "Berrypicking Model, treated in the next chapter, shares this ecological flavor but "
        "focuses on how the query itself changes during a search. For a note taking practice, "
        "the lesson is that notes should advertise their content honestly in the title, giving "
        "future readers a reliable scent to follow. Poor titles are depleted patches: they cost "
        "a visit and return nothing. Strong titles keep the cost of foraging low across an "
        "entire collection.",
    ),
    (
        "Berrypicking",
        "The Berrypicking Model describes real searches as a sequence of small pickings rather than "
        "one perfect query. In the Berrypicking Model, every retrieved document changes the "
        "searcher's understanding, and the next query reflects that change. The classic image is a "
        "person moving through bushes, taking a few berries here and a few there, never filling the "
        "basket in one place. The Berrypicking Model broke with older retrieval models that assumed "
        "a fixed information need answered by a single ranked list. Evaluations built on the older "
        "assumption underestimate how much value users draw from partial, evolving answers. The "
        "model fits everyday experience: a literature search rarely ends where it began, because "
        "each paper reshapes the question. Information Foraging Theory, described in the previous "
        "chapter, offers a complementary account of when a searcher abandons one patch for the "
        "next. Together the two frameworks explain both the path of a search and the timing of its "
        "turns. For personal knowledge management, the Berrypicking Model justifies keeping "
        "intermediate notes during a search, since the trail itself carries information. A search "
        "diary of queries and pivots often proves more valuable than the final answer, because the "
        "pivots record why the question changed.",
    ),
    (
        "Summarization",
        "Progressive Summarization is a method for condensing saved material in several passes, "
        "each pass highlighting only the best parts of the previous one. In the first pass a reader "
        "saves the raw text, in the second pass bolds the strongest sentences, and in later passes "
        "distills the bolded material further. Progressive Summarization spreads the cost of "
        "summarizing across many touchpoints instead of paying it all at once. The method is "
        "designed for opportunistic compression: a note is refined a little each time it is "
        "actually used. Critics note that Progressive Summarization can become busywork when "
        "applied to material that will never be read again. Its defenders answer that the method "
        "is explicitly demand driven, since passes happen only on notes that keep proving useful. "
        "The approach pairs naturally with the Atomicity Principle, because an atomic note gives "
        "each summary pass a clear boundary. It also respects Cognitive Load Theory, since a "
        "well distilled note presents its point with minimal extraneous load. A practical rule "
        "limits each pass to a small fraction of the text, forcing genuine selection rather than "
        "wholesale highlighting. Progressive Summarization thus turns rereading, usually a weak "
        "strategy, into an act of judgment that leaves the note better than before.",
    ),
    (
        "Imagery",
        "Dual Coding Theory proposes that people process information through two connected channels, "
        "one verbal and one visual. According to Dual Coding Theory, a concept stored as both words "
        "and imagery has two retrieval paths instead of one. This redundancy explains why a sketch "
        "next to a definition improves recall even when the sketch is crude. Dual Coding Theory has "
        "guided multimedia learning research for decades, including the finding that relevant "
        "pictures help while decorative pictures hurt. The theory links naturally to the Testing "
        "Effect, because retrieval practice can target either channel and strengthen both. Learners "
        "who combine Spaced Repetition with simple diagrams report that images return to mind "
        "before the words do. For note taking, the implication is concrete: a small drawing in the "
        "margin of an atomic note is not decoration but a second code. Knowledge workers often "
        "skip visuals because drawing feels slow, yet the time cost is repaid at every future "
        "review. Cognitive Load Theory adds a boundary condition, since a diagram that must be "
        "deciphered adds load instead of sharing it. The practical advice is to draw the "
        "relationship, not the scenery, and to label the parts with the same words the note "
        "already uses.",
    ),
    (
        "Practice",
        "The final chapter connects the preceding ideas into one working routine. A note enters the "
        "system as a rough capture, is split according to the Atomicity Principle, and is linked "
        "into the network as the Zettelkasten Method prescribes. Material worth remembering is "
        "turned into questions, and Spaced Repetition schedules the reviews while the Testing "
        "Effect does the strengthening. Reading and research follow the ecological picture drawn "
        "by Information Foraging Theory and the Berrypicking Model, with queries evolving as "
        "understanding grows. Notes that keep earning attention are compressed through Progressive "
        "Summarization, and important ones gain a sketch because Dual Coding Theory promises a "
        "second retrieval path. Cognitive Load Theory supervises the whole pipeline, vetoing any "
        "habit that adds friction without adding learning. None of these methods is exotic, and "
        "none requires special software, although software can lower the cost of each step. What "
        "the routine requires is consistency, because every component compounds: links accumulate, "
        "intervals lengthen, and summaries sharpen. A practitioner who sustains the routine for a "
        "year typically reports that old notes answer new questions, which is the entire point of "
        "the exercise. The system succeeds when finding an old thought is faster than having it "
        "again.",
    ),
]

MARGIN = 72  # 1 Zoll
PAGE_W, PAGE_H = 595, 842  # A4


def build_pdf() -> pymupdf.Document:
    doc = pymupdf.open()
    for header, body in CHAPTERS:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        # Einwort-Header in 18pt (> avg*1.15 -> pdf_chunker-Chunk-Break)
        page.insert_textbox(
            pymupdf.Rect(MARGIN, MARGIN, PAGE_W - MARGIN, MARGIN + 40),
            header,
            fontsize=18,
            fontname="hebo",
        )
        rc = page.insert_textbox(
            pymupdf.Rect(MARGIN, MARGIN + 50, PAGE_W - MARGIN, PAGE_H - MARGIN),
            body,
            fontsize=11,
            fontname="helv",
            lineheight=1.45,
        )
        if rc < 0:
            raise SystemExit(f"Kapitel '{header}': Text passt nicht auf eine Seite (rc={rc})")
    # Feste Metadaten -> keine Zeitstempel-Drift zwischen Generator-Laeufen
    doc.set_metadata({"title": "E2E Corpus (synthetic PKM primer)", "author": "atomic-notes fixtures_gen"})
    return doc


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "fixtures" / "e2e_corpus.pdf"
    out.parent.mkdir(exist_ok=True)
    doc = build_pdf()
    doc.save(str(out), deflate=True)
    print(f"E2E-Corpus ({doc.page_count} Seiten) -> {out}")


if __name__ == "__main__":
    main()
