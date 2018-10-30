#include "dqn.h"
#include <iostream>

DQN::DQN()
{
  cnn = nullptr;
}

DQN::DQN( Json::Value &json_config,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size)
    :DQNInterface(state_geometry, actions_count, experience_buffer_size)
{
  cnn = nullptr;
  init(json_config, gamma,  state_geometry, actions_count, experience_buffer_size);
}



DQN::~DQN()
{
  if (cnn != nullptr)
  {
    delete cnn;
    cnn = nullptr;
  }
}


void DQN::init(   Json::Value &json_config,
                  float gamma,
                  sGeometry state_geometry,
                  unsigned int actions_count,
                  unsigned int experience_buffer_size)
{
  init_interface(state_geometry, actions_count, experience_buffer_size);

    if (cnn != nullptr)
    {
      delete cnn;
      cnn = nullptr;
    }

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

    this->gamma = gamma;

    cnn = nullptr;

    sGeometry output_geometry;

    output_geometry.w = 1;
    output_geometry.h = 1;
    output_geometry.d = actions_count;


    cnn = new CNN(json_config, state_geometry, output_geometry);
}


void DQN::compute_q_values(std::vector<float> &state)
{
  cnn->forward(q_values, state);
}


void DQN::add(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float reward)
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

void DQN::add_final(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float final_reward)
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

void DQN::learn()
{
  int ptr = current_ptr-1;

  unsigned int state            = ptr;
  unsigned int action           = experience_buffer[state].action;
  float reward                  = experience_buffer[state].reward;

  experience_buffer[state].q_values[action] = reward;

  ptr--;

  float limit = 0.999;

  while (ptr >= 0)
  {
    unsigned int state            = ptr;
    unsigned int state_next       = ptr + 1;
    unsigned int action           = experience_buffer[state].action;
    unsigned int best_action_next = argmax(experience_buffer[state_next].q_values);
    float reward                  = experience_buffer[state].reward;

    float gamma_ = gamma;
    if (experience_buffer[state].is_final)
      gamma_ = 0.0;

    float q = reward + gamma_*experience_buffer[state_next].q_values[best_action_next];


    experience_buffer[state].q_values[action] = q;

    for (unsigned int i = 0; i < experience_buffer[state].q_values.size(); i++)
      experience_buffer[state].q_values[i] = saturate(experience_buffer[state].q_values[i], -limit, limit);

    ptr--; 
  }

  cnn->set_training_mode();

  for (unsigned int i = 0; i < current_ptr; i++)
  {
    cnn->train(experience_buffer[i].q_values, experience_buffer[i].state);
  }

  cnn->unset_training_mode();

  test();

  new_batch();
}

void DQN::test()
{
  unsigned int output_size = experience_buffer[0].q_values.size();

	compare.clear();
	compare.set_output_size(output_size);

	std::vector<float> nn_output(output_size);

	for (unsigned int i = 0; i < current_ptr; i++)
	{
		cnn->forward(nn_output, experience_buffer[i].state);
		compare.compare(experience_buffer[i].q_values, nn_output, experience_buffer[i].action);
	}

	compare.process(101);
}

void DQN::new_batch()
{
  current_ptr = 0;
  buffer_clear();
}

bool DQN::is_full()
{
  if (current_ptr >= experience_buffer.size())
    return true;

  return false;
}

void DQN::buffer_clear()
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


void DQN::print()
{
  for (unsigned int j = 0; j < current_ptr; j++)
  {
    std::cout << "[";
    std::cout << j << " ";
    std::cout << experience_buffer[j].action << " ";
    std::cout << experience_buffer[j].q_values[experience_buffer[j].action] << " ";
    std::cout << experience_buffer[j].is_final << " ";
    std::cout << "]\n";
  }

  std::cout << "\n\n";
}