import os

import cami

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_tests_import_local_source_not_installed_copy():
    package_dir = os.path.dirname(os.path.abspath(cami.__file__))
    assert os.path.normcase(package_dir) == os.path.normcase(os.path.join(REPO_ROOT, 'cami'))


PUBLIC_API = {
    'cami', 'cami_rate', 'directionality', 'entropy', 'get_prob', 'joint_entropy', 'multithread_causality',
    'multivar_joint_entropy', 'mutual_info', 'mutual_info_rate', 'pointwise', 'surrogate_test',
    'symbolic_encoding', 'total_correlation', 'transfer_entropy', 'transfer_entropy_rate',
}


def test_public_api_is_explicit():
    assert set(cami.__all__) == PUBLIC_API
    namespace = {}
    exec('from cami import *', namespace)
    assert set(namespace) - {'__builtins__'} == PUBLIC_API
    assert all(callable(getattr(cami, name)) for name in PUBLIC_API)
