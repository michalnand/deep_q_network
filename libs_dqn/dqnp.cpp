#include "dqnp.h"

#include <iostream>
#include <math.h>
#include <random_distribution.h>

DQNP::DQNP()
     :DQNInterface()
{
  cnn = nullptr;
}

DQNP::DQNP( Json::Value &json_config,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size)
    :DQNInterface(state_geometry, actions_count, experience_buffer_size)
{
  cnn = nullptr;
  init(json_config, gamma,  state_geometry, actions_count, experience_buffer_size);
}


DQNP::DQNP( std::string json_config_file_name,
            float gamma,
            sGeometry state_geometry,
            unsigned int actions_count,
            unsigned int experience_buffer_size)
    :DQNInterface(state_geometry, actions_count, experience_buffer_size)
{
  cnn = nullptr;
  JsonConfig json_config(json_config_file_name);
  init(json_config.result, gamma,  state_geometry, actions_count, experience_buffer_size);
}

DQNP::DQNP( std::string json_config_file_name,
          sGeometry state_geometry,
          unsigned int actions_count)
     :DQNInterface()
{
  cnn = nullptr;
  JsonConfig json_config(json_config_file_name);

  float gamma = json_config.result["gamma"].asFloat();
  unsigned int experience_buffer_size = json_config.result["experience_buffer_size"].asInt();

  init(json_config.result["network_architecture"], gamma, state_geometry, actions_count, experience_buffer_size);
}

DQNP::DQNP( std::string json_config_file_name)
     :DQNInterface()
{
    cnn = nullptr;
    JsonConfig json_config(json_config_file_name);

    float gamma = json_config.result["gamma"].asFloat();

    unsigned int experience_buffer_size = json_config.result["experience_buffer_size"].asInt();
    unsigned int actions_count = json_config.result["actions_count"].asInt();

    sGeometry state_geometry;
    state_geometry.w = json_config.result["state_geometry"][0].asInt();
    state_geometry.h = json_config.result["state_geometry"][1].asInt();
    state_geometry.h = json_config.result["state_geometry"][2].asInt();

    init(json_config.result["network_architecture"], gamma, state_geometry, actions_count, experience_buffer_size);
}




DQNP::~DQNP()
{
  if (cnn != nullptr)
  {
    delete cnn;
    cnn = nullptr;
  }
}


void DQNP::init(   Json::Value &json_config,
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

    priority.resize(experience_buffer_size);
    for (unsigned int i = 0; i < priority.size(); i++)
      priority[i] = 1.0;

    this->gamma = gamma;

    cnn = nullptr;

    sGeometry output_geometry;

    output_geometry.w = 1;
    output_geometry.h = 1;
    output_geometry.d = actions_count;


    cnn = new CNN(json_config, state_geometry, output_geometry);
}


void DQNP::compute_q_values(std::vector<float> &state)
{
  cnn->forward(q_values, state);
}

void DQNP::learn()
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


    float error = q - experience_buffer[state].q_values[action];

    if (error < 0)
      error = -error;

    priority[ptr] = error;

    experience_buffer[state].q_values[action] = q;

    for (unsigned int i = 0; i < experience_buffer[state].q_values.size(); i++)
      experience_buffer[state].q_values[i] = saturate(experience_buffer[state].q_values[i], -limit, limit);

    ptr--;
  }

  RandomDistribution distribution(priority, current_ptr);

  cnn->set_training_mode();

  for (unsigned int i = 0; i < current_ptr; i++)
  {
    unsigned int idx = distribution.get();
    cnn->train(experience_buffer[idx].q_values, experience_buffer[idx].state);
  }

  cnn->unset_training_mode();

  test();

  new_batch();
}

void DQNP::test()
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

void DQNP::new_batch()
{
  current_ptr = 0;
  buffer_clear();
}

bool DQNP::is_full()
{
  if (current_ptr >= experience_buffer.size())
    return true;

  return false;
}
