import re, json, pathlib

inbox = pathlib.Path("Obsidian_Vault/00-inbox/atomic-review")
results = []
for md in sorted(inbox.glob("*.md")):
    text = md.read_text(encoding="utf-8")
    m = re.search(r'source-file:\s*"([^"]+)"', text)
    if not m:
        continue
    source = m.group(1)
    # quote blocks: "> [!quote]- ... \n> „...""
    for qm in re.finditer(r"> \[!quote\]-[^\n]*\n((?:> [^\n]*\n?)+)", text):
        block = qm.group(1)
        lines = [l[2:].strip() for l in block.strip().split("\n")]
        quote_text = " ".join(lines)
        quote_text = quote_text.strip('„""').strip()
        # find page from the heading line just before
        page_m = re.search(r"S\.\s*(\d+)", qm.group(0).split("\n")[0])
        page = page_m.group(1) if page_m else None
        results.append({"note": md.name, "source": source, "page": page, "quote": quote_text})

print(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nTOTAL: {len(results)} quote blocks", flush=True)
