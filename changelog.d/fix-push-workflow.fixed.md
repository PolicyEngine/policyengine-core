Fix invalid YAML in push workflow (`with:` and `token:` on the same line) that was causing all post-merge runs to fail with zero jobs, blocking version bumps and PyPI publishes.
