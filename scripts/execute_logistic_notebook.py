"""Execute and preserve every completed cell, including diagnostics on failure."""
from pathlib import Path
from datetime import datetime
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/02_eda_logistic_context.ipynb'
nb = nbformat.read(path, as_version=4)

def progress(cell, cell_index, **kwargs):
    print(datetime.now().isoformat(timespec='seconds'), 'cell', cell_index, 'completed', flush=True)
    nbformat.write(nb, path)

client = NotebookClient(nb, timeout=2400, kernel_name='epic-eda',
                        resources={'metadata': {'path': str(root)}}, on_cell_executed=progress)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print('Notebook executed and saved.', flush=True)
