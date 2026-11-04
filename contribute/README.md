# Contribute a Notebook

This page explains how to contribute a notebook, what contributors and IRSA are each responsible for, and our guidelines for notebook format and data files.

## How to Contribute

1. **Let us know what you're thinking,** whether you have an idea for a new notebook or one that's mostly completed.
   Please post in the Caltech-IPAC `#irsa-datasci` Slack channel (preferred) or open an [issue](https://github.com/Caltech-IPAC/irsa-contributed-notebooks/issues) so we can make sure it's a good fit.

2. **Create your notebook.**

   Make a copy of [`notebook_template.md`](notebook_template.md) and replace everything in curly brackets (`{...}`), including the brackets, with your own content.
   Please don't change anything outside the curly brackets.

   See [Notebook Format](#notebook-format) for information about Markdown, and [Data Files](#data-files) if your notebook depends on data files.
   For an example of a finished notebook, see [OpenUniverse2024 Quickstart](https://github.com/Caltech-IPAC/irsa-tutorials/blob/main/tutorials/simulated-data/OpenUniverse2024/openuniverse2024_quickstart.md).

   For recommended data access methods, see the [IRSA Tutorials](https://caltech-ipac.github.io/irsa-tutorials/) notebooks that use similar data.
   Whenever possible, use the cloud copy of the data.

3. **Submit your notebook.**

   Give your notebook a short, descriptive filename and put it in the [`notebooks/`](../notebooks/) subdirectory for its mission or topic.

   Add an entry for it to the site's table of contents (TOC) in [`toc.yml`](../toc.yml).

   Open a pull request (PR) and complete the author checklist in the PR description.

   Note: We prefer PRs opened from a [fork of this repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).
   If you have trouble with that, please let us know.

4. **IRSA review.**
   An IRSA reviewer will run the notebook, including the install cell, and complete the reviewer checklist in the PR description.

5. **Release.**
   Once the PR is approved, we'll merge it, and your notebook will appear on the website the next time the site is built.

6. **Maintain your notebook.**
   You can submit updates at any time by opening a PR.
   We'll let you know about any issues or questions we receive about your notebook.
   Please respond, and update the notebook if needed so it keeps working.
   We may remove notebooks with errors or other significant issues that their authors don't address.

## Contributor and IRSA Responsibilities

Contributors are responsible for following the How to Contribute instructions above.

IRSA will:

- Render the notebooks to a website.
- Maintain the repository and website infrastructure.
- Contact notebook authors when issues are reported about their notebook.

IRSA will not (except on a case-by-case basis):

- Execute the notebooks when building the website.
- Test the notebooks after they're accepted.
- Review notebooks for scientific accuracy, data access methods, or code efficiency.

## Notebook Format

Notebooks are stored in this repo as Markdown (`.md`) files, which are easier to serve and to review in PRs.
We prefer that you submit your notebook in this format, but you can also submit it as `.ipynb` and we'll help you convert it.
For how to work with this format, including converting between `.ipynb` and `.md`, see [Working with Notebooks in Markdown Format](https://docs.fornax.sciencecloud.nasa.gov/markdown-notebooks/).

## Data Files

We prefer that data files not be added to this repository.
They are typically orders of magnitude larger than code files and can make the repository bloated and difficult to work with.
We may accept data files up to about 1 MB after review.

Store larger files somewhere they can be downloaded programmatically, and have the notebook download them.
For example, a file on Google Drive shared with "Anyone with the link" can be downloaded using [gdown](https://pypi.org/project/gdown/):

```python
import gdown

url = "https://drive.google.com/file/d/{file_id}/view?usp=sharing"
gdown.download(url=url, output="{file_name}.{extension}")
```
