// Swig header for tseytin.

%include <stdint.i>
%include <std_string.i>
%include <std_vector.i>

%module tseytin %{
    #include "autosat_tseytin.h"
%}

%include "autosat_tseytin.h"

namespace std {
    %template(vectori) vector<int>;
    %template(vectorvectori) vector<vector<int>>;
}

