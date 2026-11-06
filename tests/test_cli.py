import json
import os
import subprocess
import sys


def test_cpu_demo_avoids_model_adapter_and_gpu_kernels(tmp_path):
    program = """
import importlib.abc
import runpy
import sys

class RuntimeOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'nano_hisparse.row_kernels' or fullname.split('.')[0] == 'transformers':
            raise ImportError('CPU demo imported optional dependency: ' + fullname)

sys.meta_path.insert(0, RuntimeOnly())
sys.argv = ['nano-hisparse', 'demo', '--device', 'cpu', '--top-k', '2',
            '--prompt-tokens', '65', '--gpu-pages', '12', '--requests', '2',
            '--max-new-tokens', '3', '--output', sys.argv[1]]
runpy.run_module('nano_hisparse', run_name='__main__')
"""
    output = tmp_path / "demo.json"
    result = subprocess.run(
        [sys.executable, "-c", program, str(output)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text())
    assert report["cache"]["row_stats"]["misses"] > 0
    assert all(
        r == {"prefill_count": 1, "decode_count": 2, "generated_tokens": 3}
        for r in report["requests"].values()
    )
