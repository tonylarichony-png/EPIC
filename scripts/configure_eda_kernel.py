"""Make the registered kernel activate conda DLL paths on Windows."""
from pathlib import Path
import json
import os
import sys

prefix = Path(sys.prefix)
conda = Path(os.environ.get('CONDA_EXE', prefix.parents[1] / 'Scripts/conda.exe'))
kernel = Path(os.environ['APPDATA']) / 'jupyter/kernels/epic-eda/kernel.json'
config = json.loads(kernel.read_text(encoding='utf-8'))
config['argv'] = [str(conda), 'run', '--no-capture-output', '-p', str(prefix),
                  'python', '-X', 'utf8', '-m', 'ipykernel_launcher', '-f', '{connection_file}']
kernel.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('Configured conda-activated kernel:', kernel)
