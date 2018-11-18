#ifndef _DQN_PYTHON_H_
#define _DQN_PYTHON_H_

#include <string>
#include <vector>

#include <dqn.h>
#include <ddqn.h>
#include <dqn_compare.h>
#include <dqnp.h>
#include <random_distribution.h>

#include "dqn_python.h"

void DQNTest();


class MyString: public std::string
{
  public:
    void info()
    {

    }
};

#endif
