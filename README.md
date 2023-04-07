# FlashCam Simulations

Low-level simulation package for the FlashCam. This project is intended to make simple changes to simtelarray simulations without the need to re-simulate those files. It can also generate fake gamma showers and pulses with different NSB and noise.

**Installation**

With the package structure used here, you do not have to point Python to the location of your package. You absolutely SHOULD NOT be adding the package directory to your `$PYTHONPATH`. Instead, you can use `pip` to install it locally:
```
cd ~/path/to/new_project
pip install -e .
```
`pip` will install all the dependencies specified in the `setup.cfg` file. The `-e` flag makes the install editable which means that you do not have to install the package again and again after each change. Changes to your files inside the project folder will automatically reflect in changes on your installed package. If you are working in an interactive environment (`ipython`, `Jupyter`) you will need to re-import any modules that have changed. For example, after editing `module_x.py` you will need to do the following to have the changes available in the Python interpreter:

```
import importlib
importlib.reload(module_x)
```

To install a non-editable version, do:
```
cd ~/path/to/new_project
pip install .
```

**Testing your code**

Ideally, you should be writing tests along with the new code. To test your code, first install the test dependencies:
```
pip install -e ".[test]"
```

Then run the tests from the `new_project` directory:
```
pytest --cov=.
```

The `--cov=.` flag generates a report on how much of you code is covered by tests. Ideally this should be >80%.





