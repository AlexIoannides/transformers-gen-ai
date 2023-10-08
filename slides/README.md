# Slides

Uses Pandoc.

## Building Reveal.js Presentations

```text
pandoc -t revealjs transformers-and-llms.md -s -o Demo.html \
    --slide-level=3 \
    --highlight-style=breezedark \
    --css=cusom.css \
    --variable theme=solarized
```

```text
make build-presentation
```
