# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.4.x   | :white_check_mark: |
| < 1.4   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within NeuraLearn, please send an email to the project maintainer. All security vulnerabilities will be promptly addressed.

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Security Considerations

NeuraLearn is an educational framework designed for local use. Key security notes:

- **No network access:** The framework operates entirely locally
- **No secrets required:** No API keys, tokens, or credentials are needed
- **No file system writes:** Core framework does not write to disk (visualization optional)
- **No user input execution:** CLI commands are fixed, not user-provided code
- **Synthetic data only:** Demo uses built-in XOR dataset, no external data

## Best Practices

When using NeuraLearn:

- Install only from the official repository
- Use a virtual environment for isolation
- Do not run untrusted experiment scripts without review
- Keep your Python installation updated

## Dependency Security

Core dependencies: **None** (pure Python)

Optional dependencies:
- `matplotlib>=3.5` (visualization only)
- `pytest>=7.0` (testing only)

The framework has minimal attack surface due to zero required external dependencies.
