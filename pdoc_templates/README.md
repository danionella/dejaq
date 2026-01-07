### Building the API docs locally (pdoc)

This repository includes a small pdoc template override in `pdoc_templates/`.
It expands inherited class members so subclasses show inherited method signatures
and docstrings (not just a link list).

```bash
pip install pdoc
pdoc -t pdoc_templates dejaq -o docs -d google
```
