<!--
This template is for PRs that add a new notebook.
See https://github.com/Caltech-IPAC/irsa-contributed-notebooks/blob/main/contribute/README.md for details.

For other changes, delete this template and describe your change instead.
-->

**Notebook Name:** <!-- name -->

**Summary:** <!-- one or two sentences on what the notebook does -->

## Author Checklist

- [ ] The notebook has a descriptive title and clear section names.
- [ ] Notebook authors are listed in the metadata at the top of the file.
- [ ] The first section is "Overview" and lists about 3 of the most interesting or important things the notebook does.
- [ ] The first code cell is a commented-out `%pip install` cell that installs every library the notebook imports directly, each pinned to an exact version (`==`) known to work.
- [ ] After installing those libraries, the notebook runs end-to-end without error.
- [ ] Code cells are accompanied by narrative text that explains what the code does and why.
- [ ] I added an entry for the notebook to this repository's table of contents.

---------

*To be completed by IRSA.*

## IRSA Reviewer Checklist

- [ ] The notebook meets all items in the author checklist above.
- [ ] The notebook is in `.md` format (if the author submitted `.ipynb`, help them convert it).
- [ ] All curly-bracket (`{...}`) placeholders from the template have been replaced with appropriate content.
- [ ] The "About This Notebook" section keeps the template's text about contributed notebooks and reporting issues.
- [ ] In a fresh environment, I uncommented and ran the `%pip install` cell, then ran the whole notebook without error.
- [ ] The notebook is in the appropriate directory and has a short, descriptive filename.
- [ ] Any data files added are about 1 MB or smaller and justified.
- [ ] The `toc.yml` entry is in the right section and has a short, descriptive title.
- [ ] The notebook renders correctly in the website preview.
