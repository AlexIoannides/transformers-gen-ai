# Presentation Slides

This directory contains the source material for a slide-based presentation of the work contained in this repo. This is written in markdown and rendered to a RevealJS HTML presentation using [Pandoc](https://pandoc.org).

## Building Reveal.js Presentations

The build commands for thie presentation are defined in the `Makefile`. To build them run,

```text
make build-presentation
```

### Generating PDF Versions

Open the HTML file in a broswer and then append `?print-pdf` to the URL to yield a PDF-printer friendly version of the slides.
