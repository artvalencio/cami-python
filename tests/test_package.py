import os

import cami

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_tests_import_local_source_not_installed_copy():
    package_dir = os.path.dirname(os.path.abspath(cami.__file__))
    assert os.path.normcase(package_dir) == os.path.normcase(os.path.join(REPO_ROOT, 'cami'))
