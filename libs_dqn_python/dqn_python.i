/* File : example_lib.i */
%module dqn

%include <std_string.i>
%include <std_vector.i>

using std::string;

%template(CharVector) std::vector<char>;
%template(UnsignedCharVector) std::vector<unsigned char>;
%template(IntVector) std::vector<int>;
%template(UnsignedIntVector) std::vector<unsigned int>;
%template(FloatVector) std::vector<float>;
%template(DoubleVector) std::vector<double>;

%template(IntMatrix) std::vector<std::vector<int>>;
%template(UnsignedIntMatrix) std::vector<std::vector<unsigned int>>;

%template(FloatMatrix) std::vector<std::vector<float>>;
%template(DoubleMatrix) std::vector<std::vector<double>>;


%{
#include <vector>
#include <string>

#include <nn_struct.h>
#include <log.h>


#include <dqn_interface.h>

#include <dqn_interface.h>
#include <dqn.h>
#include <ddqn.h>
#include <dqn_compare.h>
#include <dqnp.h>
#include <random_distribution.h>

#include "dqn_python.h"
%}

%include <nn_struct.h>
%include <log.h>

%include <dqn_interface.h>
%include <dqn.h>
%include <ddqn.h>
%include <dqn_compare.h>
%include <dqnp.h>
%include <random_distribution.h>

%include "dqn_python.h"
