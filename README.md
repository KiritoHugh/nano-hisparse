# nano-hisparse

A small inference engine for decoding with partially resident KV caches.

The runtime separates stable host-backed history from reusable GPU blocks.
Python 3.10+ and PyTorch 2.5+ are required; the reference path runs on CPU.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Licensed under [MIT](LICENSE).
