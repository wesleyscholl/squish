# Getting Support

## Documentation

The fastest way to find answers is the [official documentation](https://wesleyscholl.github.io/squish):

- [Installation guide](https://wesleyscholl.github.io/squish/install/)
- [Quickstart](https://wesleyscholl.github.io/squish/quickstart/)
- [API reference](https://wesleyscholl.github.io/squish/api/)
- [Architecture overview](https://wesleyscholl.github.io/squish/ARCHITECTURE/)

---

## Community

| Channel | Use for |
|---|---|
| [GitHub Discussions](https://github.com/wesleyscholl/squish/discussions) | Questions, ideas, show-and-tell |
| [GitHub Issues](https://github.com/wesleyscholl/squish/issues) | Bug reports and feature requests |
| Discord (link in README) | Real-time chat with maintainers and the community |

---

## Common Issues

**`squish: command not found`**  
Make sure the pip scripts directory is in your `PATH`. For a virtual environment, activate it first (`source .venv/bin/activate`). For Homebrew, run `brew link squish`.

**Model download is slow or interrupted**  
Run `squish pull <model>` again — downloads resume automatically.

**`ImportError: No module named 'mlx'`**  
MLX only runs on Apple Silicon. Squish requires macOS 13+ on M1–M5 hardware.

**Server not responding**  
Check that the server started with `squish run <model>` before calling the API, or use `squish daemon status`.

**Out of memory during inference**  
Try a smaller model (e.g. `llama3.1:3b` instead of `llama3.1:8b`) or close other applications to free unified memory.

---

## Security Vulnerabilities

Please **do not** open a public GitHub issue for security vulnerabilities.  
Report them privately via [GitHub Security Advisories](https://github.com/wesleyscholl/squish/security/advisories/new).

---

## Contacting the Maintainer

For anything not covered above, open a [GitHub Discussion](https://github.com/wesleyscholl/squish/discussions) or a [GitHub Issue](https://github.com/wesleyscholl/squish/issues).
