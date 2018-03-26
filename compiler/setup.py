from distutils.core import setup


from distutils.command.build_py import build_py

setup(name="Kaena-compiler",
      version="1.0",
      description="Compiler that generates Kaena-elf file from a NN",
      cmdclass={'build_py': build_py},
      packages=['tffe'],
      scripts = ['scripts/tffe'],
)
