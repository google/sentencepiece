py_ver = 3.4

all: _spm.so

spm_wrap.cxx: spm.i
	swig $(SWIGFLAGS) -python -py3 -c++ spm.i

spm_wrap.o: spm_wrap.cxx
	g++ -I. -Isrc -I/usr/include/python$(py_ver) -std=c++11 -Wall -O3 -c spm_wrap.cxx -DPIC -fPIC

_spm.so: spm_wrap.o
	g++ -shared spm_wrap.o -o _spm.so -fPIC -Lsrc/.libs -lsentencepiece -lprotobuf 

install: _spm.so
	@cp -f _spm.so $(HOME)/lib/python$(py_ver)/site-packages/
	@cp -f spm.py $(HOME)/lib/python$(py_ver)/site-packages/

clean:
	rm -f *.o _*.so *.pyc spm.py *_wrap.* *~
