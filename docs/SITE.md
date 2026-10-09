# SAVER project website

Live project page: https://wenfangsun.cn/SAVER/

The static website source is in `docs/site/`. It uses plain HTML, CSS, and JavaScript with no build dependencies. Preview it from the repository root:

```bash
python3 -m http.server 8000 --directory docs/site
```

Open http://localhost:8000/. The HTTP server is needed for the result-data fetch.

## Content and provenance

- Author names, abstract, method description, and qualitative example are from the manuscript supplied on 2026-10-07.
- `results.json` contains the two Controlled Frame-Rate Evaluation appendix tables in `sections/7_appendix.tex`: 30 temporal grounding rows and 30 video-QA rows. Each row records model size, model, fps, average observed frames, individual benchmark scores, per-benchmark frame counts, and the manuscript's macro-average. The five rates are 0.1, 0.2, 0.5, 1.0, and 2.0 fps. The 36 overlapping main-table rows were verified to match.
- Grounding value order: Charades-STA, ActivityNet, NExT-GQA, macro-average.
- QA value order: Video-MME, MVBench, LongVideoBench, MMVU, VideoMMMU, MVP, macro-average.
- Images are rendered from the supplied `framework_save.pdf` and `analysis_saver.pdf`. Original video frames belong to their respective benchmark providers.
- The Paper button and BibTeX cite arXiv:2610.10893 (cs.CV), submitted 7 October 2026. No conference acceptance is claimed.

## Publishing

GitHub Pages is configured to publish from the root of `gh-pages`. After website edits are merged into `main`, publish only the site folder:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git subtree split --prefix=docs/site -b site-publish
git push origin site-publish:gh-pages
git branch -D site-publish
```

The initial `gh-pages` commit is a subtree split from this same source, so later splits can be fast-forwarded. Do not force-push if another deployment has advanced the branch; fetch and reconcile first. GitHub Pages rebuilds after a push to `gh-pages`.

## Practical efficiency section

Selected Table 8 results are from Practical Efficiency Profiling in the supplied manuscript: Qwen3.5-2B, 2.0 versus 0.1 fps, one 96-GB H100, BF16, batch size 1. Cards use the manuscript-reported macro-average reductions (86.8% tokens, 49.3% memory, 63.6% E2E latency); displayed macro-average values are rounded, so recomputing percentages from them may differ slightly. The table shows Video-MME, LongVideoBench, VideoMMMU, and the nine-benchmark macro-average. These are base-model measurements, not a SAVER-specific speedup.

The linked VIS Lab logo is rendered from `assets/vislab-logo.pdf` in the supplied manuscript archive. The headline summarizes the five controlled evaluation frame rates, not a uniform frame-reduction claim.

## Method animation and result selection

The three-phase animated schematic replaces the method prose cards. It illustrates paired training views, the relative grounding-reward gate, and sparse-only inference. Sample marks and interval bars are illustrative, not measured examples. Playback pauses outside the viewport or when the page is hidden; users can pause or select any phase. Reduced-motion preferences disable autoplay and decorative motion.

The headline reports all five appendix frame-rate settings. The main result controls intentionally show only 0.1, 0.5, and 2.0 fps; the downloadable data retains all five settings.
