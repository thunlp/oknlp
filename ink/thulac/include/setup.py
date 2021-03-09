from distutils.core import setup, Extension

def main():
    setup(name="thulac",
          version="1.0.0",
          description="Python interface for segmentaion C++ library function",
          author="luyanjun",
          author_email="luyanjun1116@gmail.com",
          ext_modules=[Extension("thulac", ["thulac_ext.cc"])])

if __name__ == "__main__":
    main()
