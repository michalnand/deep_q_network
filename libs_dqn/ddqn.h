#ifndef _DDQN_H_
#define _DDQN_H_

#include "dqn_interface.h"


class DDQN: public DQNInterface
{
  protected:
    std::vector<float> priority;

    std::vector<float> nn_output;

  public:
    DDQN();
    DDQN(  Json::Value &json_config,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size);

    DDQN( std::string json_config_file_name,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size);

    DDQN( std::string json_config_file_name,
          sGeometry state_geometry,
          unsigned int actions_count);

    DDQN( std::string json_config_file_name);

    virtual ~DDQN();

    void init(  Json::Value &json_config,
                float gamma,
                sGeometry state_geometry,
                unsigned int actions_count,
                unsigned int experience_buffer_size);


    void compute_q_values(std::vector<float> &state);

    void learn();
    void test();

    void new_batch();

    bool is_full();

  protected:
    void q_values_to_nn_output(std::vector<float> &nn_output, std::vector<float> &q_values);
    void nn_output_to_q_values(std::vector<float> &q_values, std::vector<float> &nn_output);
    float v_average(std::vector<float> &v, int size = -1);

};

#endif
