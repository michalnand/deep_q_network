/* File : example_lib.i */
%module dqn

%include <std_vector.i>
%include <std_string.i>

%template(VectorChar) std::vector<char>;
%template(VectorUnsignedChar) std::vector<unsigned char>;
%template(VectorInt) std::vector<int>;
%template(VectorUnsignedInt) std::vector<unsigned int>;
%template(VectorFloat) std::vector<float>;
%template(VectorDouble) std::vector<double>;

%template(MatrixInt) std::vector<std::vector<int>>;
%template(MatrixUnsignedInt) std::vector<std::vector<unsigned int>>;

%template(MatrixFloat) std::vector<std::vector<float>>;
%template(MatrixDouble) std::vector<std::vector<double>>;


%module example

%apply const std::string& {std::string* foo};



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
