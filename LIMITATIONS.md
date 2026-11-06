# Technical Limitations & Future Engineering Roadmap

1. **Multi-Page Financial Tables**: Large SEC 10-K tables spanning across page breaks require manual chunk stitching.
2. **Context Window Scaling**: Lightweight local embeddings (all-MiniLM-L6-v2) prioritize zero-cost offline speed over ultra-long context windows.
3. **Cross-Filing Comparison**: Currently queries single 10-K filings. Multi-company comparative reporting is planned for future iterations.
