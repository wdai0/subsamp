"""Check release contents against the checkout: python tests/check_distributions.py dist."""

import argparse
import json
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check(directory):
    directory = Path(directory)
    wheels, sources = list(directory.glob('*.whl')), list(directory.glob('*.tar.gz'))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError('Expected exactly one wheel and one source distribution')
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_files = {name for name in archive.namelist() if not name.endswith('/')}
        if any(not (name.startswith('subsampwinner/') or
                    name.split('/')[0].startswith('subsampwinner-') and
                    name.split('/')[0].endswith('.dist-info')) for name in wheel_files):
            raise ValueError('Wheel contains files outside the runtime package and metadata')
        for source in sorted((ROOT / 'src/subsampwinner').glob('*.py')):
            name = str(source.relative_to(ROOT / 'src'))
            if archive.read(name) != source.read_bytes():
                raise ValueError(f'Wheel source differs: {name}')
    with tarfile.open(sources[0], 'r:gz') as archive:
        files = {}
        for member in archive.getmembers():
            if member.isfile():
                relative = member.name.split('/', 1)[1]
                files[relative] = archive.extractfile(member).read()
        if any(name.startswith(('output/', '.git/', '.venv/', 'build/', 'dist/')) or
               '__pycache__' in Path(name).parts or name.endswith(('.pyc', '.pyo')) for name in files):
            raise ValueError('Source distribution contains generated output or local build state')
        required = [ROOT / name for name in ('README.md', 'LICENSE', 'setup.py', 'pyproject.toml', 'MANIFEST.in')]
        for directory_name, suffixes in (('src/subsampwinner', {'.py'}),
                                         ('tests', {'.py', '.R', '.json', '.npz', '.txt'}),
                                         ('benchmarks', {'.py', '.json', '.md'}),
                                         ('docs', {'.md', '.txt', '.json'})):
            required.extend(path for path in (ROOT / directory_name).rglob('*') if path.is_file() and path.suffix in suffixes)
        for source in required:
            name = str(source.relative_to(ROOT))
            if files.get(name) != source.read_bytes():
                raise ValueError(f'Source distribution is missing or differs from: {name}')
    return dict(wheel=dict(name=wheels[0].name, bytes=wheels[0].stat().st_size, files=len(wheel_files)),
                sdist=dict(name=sources[0].name, bytes=sources[0].stat().st_size, files=len(files)),
                status='Runtime-only wheel; complete source, tests, fixtures, and benchmark helpers; no generated outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(json.dumps(check(parser.parse_args().directory), indent=2))
