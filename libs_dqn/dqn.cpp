#include "dqn.h"
#include <iostream>

DQN::DQN()
    :DQNInterface()
{
  cnn = nullptr;
}

DQN::DQN( Json::Value &json_config,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size,
          bool normalise)
    :DQNInterface(state_geometry, actions_count, experience_buffer_size, normalise)
{
  cnn = nullptr;
  init(json_config, gamma,  state_geometry, actions_count, experience_buffer_size, normalise);
}

DQN::DQN( std::string json_config_file_name,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size,
          bool normalise)
    :DQNInterface(state_geometry, actions_count, experience_buffer_size, normalise)
{
  cnn = nullptr;
  JsonConfig json_config(json_config_file_name);
  init(json_config.result, gamma,  state_geometry, actions_count, experience_buffer_size, normalise);
}

DQN::DQN( std::string json_config_file_name,
          sGeometry state_geometry,
          unsigned int actions_count)
    :DQNInterface()
{
  cnn = nullptr;
  JsonConfig json_config(json_config_file_name);

  float gamma = json_config.result["gamma"].asFloat();
  unsigned int experience_buffer_size = json_config.result["experience_buffer_size"].asInt();
  bool normalise = json_config.result["normalise"].asBool();

  init(json_config.result["network_architecture"], gamma, state_geometry, actions_count, experience_buffer_size, normalise);
}

DQN::DQN(std::string json_config_file_name)
    :DQNInterface()
{
    cnn = nullptr;
    JsonConfig json_config(json_config_file_name);

    float gamma = json_config.result["gamma"].asFloat();

    unsigned int experience_buffer_size = json_config.result["experience_buffer_size"].asInt();
    unsigned int actions_count = json_config.result["actions_count"].asInt();
    bool normalise = json_config.result["normalise"].asBool();

    sGeometry state_geometry;
    state_geometry.w = json_config.result["state_geometry"][0].asInt();
    state_geometry.h = json_config.result["state_geometry"][1].asInt();
    state_geometry.h = json_config.result["state_geometry"][2].asInt();

    init(json_config.result["network_architecture"], gamma, state_geometry, actions_count, experience_buffer_size, normalise);
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
                  unsigned int experience_buffer_size,
                  bool normalise)
{
    init_interface(state_geometry, actions_count, experience_buffer_size, normalise);

    if (cnn != nullptr)
    {
      delete cnn;
      cnn = nullptr;
    }

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

void DQN::learn()
{
    int ptr = current_ptr-1;

    unsigned int state            = ptr;
    unsigned int action           = experience_buffer[state].action;
    float reward                  = experience_buffer[state].reward;

    experience_buffer[state].q_values[action] = reward;

    ptr--;

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
        ptr--;
    }

    if (normalise)
        experience_buffer_normalise();

    experience_buffer_clip();

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
