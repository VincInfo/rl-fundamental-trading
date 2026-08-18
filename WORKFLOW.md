# Workflow Guidelines

To keep our code clean, avoid merge conflicts, and make tracking our work easy, please follow this (simple) workflow.

## 1. Issues First

- **Always create an Issue** before starting work on any new feature, bug fix, refactoring, or any other change.
- Clearly describe the task in the issue description.
- Assign the issue to yourself once you start working on it.

## 2. Branching from Issues
* Create a dedicated branch directly from the issue using **GitHub's built-in feature** (see [GitHub Docs](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-a-branch-for-an-issue#creating-a-branch-for-an-issue)):
  * Open the issue and look at the right sidebar under _**Development**_.
  * Click _**Create a branch**_.

  <img width="260" style="border: 1px solid #e1e4e8; border-radius: 6px; margin: 8px 0;" alt="Create a branch button in Development sidebar" src="https://github.com/user-attachments/assets/731c3b7a-b753-4c22-83df-b0aae4a16fdd" />

* **Traceability:** Creating the branch directly from the issue automatically links the two together, which ensures clear traceability
* **Naming Format:** `<issue-id>-<issue-title>` (_this is done automatically if you use GitHub's built-in feature_)
* **Examples:**
  - `3-hybrid-model-architecture-documentation`
  - `5-add-workflow-and-branching-guidelines`

***Important note**: Avoid working directly on `main`.*

## 3. Commit Messages

- Start with a capitalized imperative verb (e.g., `Add`, `Fix`, `Update`, `Remove`, `Refactor`).
- Keep it short and simple.
- **Examples:**
  - `Add user registration endpoint`
  - `Fix login validation error`
  - `Update database for profiles`
  - `Refactor navigation bar component`

## 4. Pull Requests (PRs)

- **Link the Issue:** Explicitly link the PR to the relevant issue so it automatically closes when merged (see [GitHub Docs: Linking a pull request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)):
  - Use closing keywords in the PR description: `Closes #12`, `Fixes #45`, or `Resolves #78` (_see doc above for more keywords_)
  - Alternatively, link it manually in the right sidebar under _**Development**_.

- **Quick Summary:** Add 1–2 sentences explaining what was changed.
- **Review:** Request **at least one code review** from a teammate before merging.
- **Merge Strategy:** Prefer **Squash and Merge** to keep the commit history clean and merge into `main` once approved.
