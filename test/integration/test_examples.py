import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


@pytest.mark.parametrize("notebook_file", ["examples/1-transformer-from-scratch.ipynb"])
def test_notebook_execution(notebook_file):
    with open(notebook_file, "r") as f:
        notebook_content = nbformat.read(f, as_version=4)

    executor = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        executor.preprocess(notebook_content, {"metadata": {"path": "./"}})
    except Exception as e:
        pytest.fail(f"Error in notebook execution: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
