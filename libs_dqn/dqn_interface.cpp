#include "dqn_interface.h"


DQNInterface::DQNInterface()
{

}

DQNInterface::DQNInterface(sGeometry state_geometry, unsigned int actions_count, unsigned int experience_buffer_size)
{
  init_interface(state_geometry, actions_count, experience_buffer_size);
}

DQNInterface::~DQNInterface()
{

}


void DQNInterface::init_interface(sGeometry state_geometry, unsigned int actions_count, unsigned int experience_buffer_size)
{
  this->state_geometry = state_geometry;
  this->actions_count = actions_count;
  this->experience_buffer_size = experience_buffer_size;

  q_values.resize(actions_count);

  this->state_size = state_geometry.w*state_geometry.h*state_geometry.d;
}

std::vector<float>& DQNInterface::get_q_values()
{
  return q_values;
}

float DQNInterface::get_max_q_value()
{
  return q_values[argmax(q_values)];
}


void DQNInterface::compute_q_values(std::vector<float> &state)
{
  (void)state;
}

void DQNInterface::add(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float reward)
{
  (void)state;
  (void)q_values;
  (void)action;
  (void)reward;
}

void DQNInterface::add_final(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float final_reward)
{
  (void)state;
  (void)q_values;
  (void)action;
  (void)final_reward;
}

void DQNInterface::learn()
{

}

void DQNInterface::new_batch()
{

}

bool DQNInterface::is_full()
{
  return false;
}

void DQNInterface::test()
{

}

DQNCompare& DQNInterface::get_compare_result()
{
  return compare;
}

float DQNInterface::saturate(float value, float min, float max)
{
  if (value > max)
    value = max;
  if (value < min)
    value = min;

  return value;
}

unsigned int DQNInterface::argmax(std::vector<float> &v)
{
  unsigned int result = 0;
  for (unsigned int i = 0; i < v.size(); i++)
    if (v[i] > v[result])
      result = i;

  return result;
}
