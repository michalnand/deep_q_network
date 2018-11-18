#ifndef _DQN_H_
#define _DQN_H_

#include "dqn_interface.h"


class DQN: public DQNInterface
{
  protected:
    float gamma;

  public:
    DQN();
    DQN(  Json::Value &json_config,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size);

    DQN(  std::string json_config_file_name,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size);

    DQN( std::string json_config_file_name,
         sGeometry state_geometry,
         unsigned int actions_count); 

    virtual ~DQN();

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
};

#endif
