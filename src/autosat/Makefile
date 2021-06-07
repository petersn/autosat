
CFLAGS=-O3 -flto -g -fPIC -Wall -Wextra -Wpedantic -Wno-sign-compare `pkg-config --cflags --libs python3` -fdiagnostics-color
CXXFLAGS=$(CFLAGS) -std=c++17

all: _autosat_tseytin.so

autosat_tseytin_wrap.cxx: autosat_tseytin.h autosat_tseytin.i Makefile
	swig -c++ -python autosat_tseytin.i

%.o: %.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $^

_autosat_tseytin.so: autosat_tseytin.o autosat_tseytin_wrap.o
	$(CXX) -shared -Wl,-soname,$@ $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o *.pyc autosat_tseytin_wrap.cxx _autosat_tseytin.so autosat_tseytin.py

