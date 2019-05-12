# Keywords extraction library

Extract keywords using different algorithms and sci-kit learn interfaces

## Example

```python
from keyverbum.keywords import TfIdf, StopwordsFilter

text = (
    "Inverse problems for a mathematical model of ion exchange in a "
    "compressible ion exchanger. A mathematical model of ion exchange "
    "is considered, allowing for ion exchanger compression in the process "
    "of ion exchange. Two inverse problems are investigated for this model, "
    "unique solvability is proved, and numerical solution methods are proposed. "
    "The efficiency of the proposed methods is demon strated by a numerical experiment"
)

stop_words = StopwordsFilter('english')
key_extract = TfIdf(tokens_filter=stop_words).fit(text)
key_extract.predict(text)
# ["ion", "of"]
```
