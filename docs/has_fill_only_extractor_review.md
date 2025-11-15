# HAS Fill-only Extractor v2.3d – Code Review

## Overall quality
- **Monolithic script** – Everything lives in the module namespace without abstraction. Responsibilities such as PDF text extraction, section parsing, and spreadsheet mutation are tangled together, which makes the control flow difficult to follow and hard to test.
- **Heavy, repeated I/O** – Each spreadsheet row re-reads and re-normalises PDF files although many rows share the same CT identifier. The lack of caching makes the runtime explode on large batches.
- **Minimal diagnostics** – Console output is limited to the final file paths. When a PDF cannot be parsed you only discover it in the CSV logs afterwards, which complicates debugging.
- **Regex handling is brittle** – Patterns are re-compiled on every lookup. Additionally, the section slicing strategy (`split("\n\n")[0]`) discards legitimate paragraphs whenever the source layout uses single newlines.
- **Missing typing and structure** – No type hints (except on helper functions) or data classes describe the format of targets, diagnostics, or unmatched rows, so accidental schema changes are easy to miss.

## Functional issues discovered
1. **Expensive repeated PDF parsing.** Every spreadsheet row iterates over the CT's PDF candidates and calls `extract_text_any` even if another row already processed the same file. For submissions that reuse CTs across strengths or pack sizes this produces quadratic work and can easily burn minutes.
2. **Incorrect section truncation.** `extract_section` narrows the body to `snippet = after.strip().split("\n\n")[0]`. PDFs that use single line breaks never expose a blank line, so the function returns the entire remainder of the document until a stop token happens to match, often filling the SSOT columns with multiple sections at once.
3. **Opaque diagnostics for extraction failures.** When neither pdfminer nor PyPDF2 returns text the script silently sets `text = ""`, logs a row in `diag_rows`, and moves on. There is no log-level feedback, and the CSV row merely says `no_text`, so you cannot differentiate between parser crashes, encrypted PDFs, or empty files while the script is still running.

The new implementation in `scripts/has_fill_only_extractor_v2_4.py` addresses these problems by introducing caching, structured extractors, reliable section slicing, and verbose logging.
