# Development Guidelines

## 1. Architecture

- Maintain clear separation between game logic, AI agents, evaluation, and UI.
- Keep the game engine independent of rendering and visualization.
- Prefer modular, maintainable, and testable designs.
- Avoid unnecessary abstractions and architectural complexity.

## 2. Development Workflow

- Inspect relevant existing code before implementing changes.
- For substantial features or architectural changes, present a plan and wait for approval.
- Explain important design decisions and trade-offs.
- Prefer incremental changes over large rewrites.

## 3. Code Quality

- Follow existing Python conventions.
- Use type hints where appropriate.
- Avoid unnecessary dependencies.
- Consider performance, maintainability, and extensibility.
## Code Comments

- Write code comments in Turkish
- Start comments with lowercase letters
- Do not use punctuation at the end of comments
- Keep technical terms in English
- Add comments only when they explain non-obvious logic or important design decisions
- Do not add comments that merely restate the code
- Keep identifiers, docstrings, type hints, and error messages in English
- Preserve existing comments unless modification is necessary
- Do not use turkish characters 

## 4. Testing

- Add tests for new game mechanics.
- Run relevant pytest tests after implementation.
- Report failures clearly.
- Never modify tests merely to conceal implementation bugs.

## 5. Git Safety

- Never commit, push, reset, rebase, or discard changes without explicit approval.
- Preserve all existing uncommitted work.
- Review the Git diff after modifications.

## 6. AI Development

- Keep AI agents independent from game rules and rendering.
- Preserve compatibility with the existing Gymnasium environment.
- Design experiments to support reproducible benchmarking.

## 7. Collaboration

- Explain what you intend to change before substantial implementations.
- Ask for clarification when requirements are ambiguous.
- Do not introduce unrelated changes.
