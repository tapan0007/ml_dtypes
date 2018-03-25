from distutils.core import setup
from distutils.command.build_py import build_py


setup(name="kaena_runtime",
      version="1.0",
      description="kaena runtime package.",
      cmdclass={'build_py': build_py},
      packages=[],
      scripts = ['util/nn_executor', 'util/runtime_tf', 'util/runtime_sim'],
)
