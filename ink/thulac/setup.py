from distutils.core import setup, Extension
import os
path = os.getcwd()
os.environ['CC']='g++'
def main():
    setup(name="lacthu",
          version="1.0.0",
          description="Python interface for segmentaion C++ library function",
          author="luyanjun",
          author_email="luyanjun1116@gmail.com",
          ext_modules=[
              Extension("lacthu",
             extra_compile_args=["-O2","-Wall","-std=c++11"],
              sources = ["thulac_cls.cc"],
              include_dirs = [os.path.join(path,'include')],
              )])

if __name__ == "__main__":
    main()
