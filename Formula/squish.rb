class Squish < Formula
  desc "Run 70B language models on a MacBook — memory-mapped INT8 inference via Apple MLX"
  homepage "https://wesleyscholl.github.io/squish"
  # HEAD-only until the v1.0.0 release tag is pushed.
  # Install with: brew install --HEAD wesleyscholl/squish/squish
  # Once v1.0.0 is tagged, uncomment and fill:
  #   url "https://github.com/wesleyscholl/squish/archive/refs/tags/v1.0.0.tar.gz"
  #   sha256 "<sha256-of-release-tarball>"
  license "MIT"
  head "https://github.com/wesleyscholl/squish.git", branch: "main"

  depends_on "python@3.12"
  depends_on :macos
  depends_on arch: :arm64    # Apple Silicon (M1–M5) required

  def install
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install_and_link buildpath
    bin.install_symlink libexec/"bin/squish"
  end

  def caveats
    <<~EOS
      Squish requires Apple Silicon (M1 or later) and macOS 13 Ventura+.
      Models are stored in ~/.squish/models/ by default.

      Get started:
        squish pull qwen3:8b
        squish serve qwen3:8b

      OpenAI-compatible API:
        curl http://localhost:11435/v1/chat/completions \\
          -H "Content-Type: application/json" \\
          -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"Hello!"}]}'
    EOS
  end

  test do
    assert_match "squish 1.0.0", shell_output("#{bin}/squish --version")
  end
end
