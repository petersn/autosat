
CFLAGS=-O3 -flto -g -fPIC -Wall -Wextra -Wpedantic -Wno-sign-compare `pkg-config --cflags --libs python3` -fdiagnostics-color
CXXFLAGS=$(CFLAGS) -std=c++17

all: _tseytin.so

tseytin_wrap.cxx: tseytin.h tseytin.i Makefile
	swig -c++ -python tseytin.i

%.o: %.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $^

_tseytin.so: tseytin.o tseytin_wrap.o
	$(CXX) -shared -Wl,-soname,$@ $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o *.pyc tseytin_wrap.cxx _tseytin.so tseytin.py

