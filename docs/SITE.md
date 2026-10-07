# SAVER project website

Live project page: https://lmsdss.github.io/SAVER/

The static website source is in `docs/site/`. It uses plain HTML, CSS, and JavaScript with no build dependencies. Preview it from the repository root:

```bash
python3 -m http.server 8000 --directory docs/site
```

Open http://localhost:8000/. The HTTP server is needed for the result-data fetch.

## Content and provenance

- Author names, abstract, method description, and qualitative example are from the manuscript supplied on 2026-10-07.
- `results.json` contains the controlled evaluation blocks of the two main tables in `sections/5_experiments.tex`: 18 temporal grounding rows and 18 video-QA rows. Each row records model size, model, fps, average observed frames, individual benchmark scores, and the manuscript's macro-average.
- Grounding value order: Charades-STA, ActivityNet, NExT-GQA, macro-average.
- QA value order: Video-MME, MVBench, LongVideoBench, MMVU, VideoMMMU, MVP, macro-average.
- Images are rendered from the supplied `framework_save.pdf` and `analysis_saver.pdf`. Original video frames belong to their respective benchmark providers.
- No acceptance venue or arXiv identifier is claimed. The BibTeX entry is a project citation for the current manuscript. Update it and add a Paper button once the final publication URL is available.

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
