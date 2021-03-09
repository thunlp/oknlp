from distutils.core import setup, Extension
import os
path = os.getcwd()
os.environ['CC']='g++'
def main():
    setup(name="thulac",
          version="1.0.0",
          description="Python interface for segmentaion C++ library function",
          author="luyanjun",
          author_email="luyanjun1116@gmail.com",
          ext_modules=[
              Extension("thulac",
              extra_compile_args=["-O3","-Wall","-std=c++11"],
              sources = ["thulac_ext.cc"],
              include_dirs = [os.path.join(path,'include')],
              )])

if __name__ == "__main__":
    main()
