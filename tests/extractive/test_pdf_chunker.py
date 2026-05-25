from extractive.pipeline.pdf_chunker import detect_language, strip_anchors

def test_detect_english():
    assert detect_language("This is an English text about information behavior.") == "en"

def test_detect_german():
    lang = detect_language("Das ist ein deutscher Text ueber Informationsverhalten.")
    assert lang == "de"

def test_strip_anchors_removes_marker():
    assert strip_anchors("IB is broad. (S. 3)") == "IB is broad."

def test_strip_anchors_no_change():
    assert strip_anchors("No anchor here.") == "No anchor here."

def test_strip_anchors_range():
    assert strip_anchors("Something. (S. 3-4)") == "Something."
