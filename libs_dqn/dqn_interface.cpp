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


  state_size = state_geometry.w*state_geometry.h*state_geometry.d;

  experience_buffer.resize(experience_buffer_size);

  for (unsigned int i = 0; i < experience_buffer.size(); i++)
  {
    experience_buffer[i].state.resize(state_size);
    experience_buffer[i].q_values.resize(actions_count);

    experience_buffer[i].reward   = 0.0;
    experience_buffer[i].action   = 0;
    experience_buffer[i].is_final = false;
  }


  current_ptr = 0;
  buffer_clear();
}


void DQNInterface::buffer_clear()
{
  for (unsigned int j = 0; j < experience_buffer.size(); j++)
  {
    for (unsigned int i = 0; i < state_size; i++)
      experience_buffer[j].state[i] = 0.0;

    for (unsigned int i = 0; i < actions_count; i++)
      experience_buffer[j].q_values[i] = 0.0;

    experience_buffer[j].action   = 0;
    experience_buffer[j].reward   = 0.0;
    experience_buffer[j].is_final = false;
  }
}

void DQNInterface::add(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float reward)
{
  if (current_ptr < experience_buffer.size())
  {
    experience_buffer[current_ptr].state      = state;
    experience_buffer[current_ptr].q_values   = q_values;
    experience_buffer[current_ptr].action     = action;
    experience_buffer[current_ptr].reward     = reward;
    experience_buffer[current_ptr].is_final   = false;

    current_ptr++;
  }
}

void DQNInterface::add_final(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float final_reward)
{
  if (current_ptr < experience_buffer.size())
  {
    experience_buffer[current_ptr].state      = state;
    experience_buffer[current_ptr].q_values   = q_values;
    experience_buffer[current_ptr].action     = action;
    experience_buffer[current_ptr].reward     = final_reward;
    experience_buffer[current_ptr].is_final   = true;

    current_ptr++;
  }
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

void DQNInterface::save(std::string file_name_prefix)
{
  cnn->save(file_name_prefix);
}

void DQNInterface::load_weights(std::string file_name_prefix)
{
  cnn->load_weights(file_name_prefix);
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
