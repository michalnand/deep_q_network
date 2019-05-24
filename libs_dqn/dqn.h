#ifndef _DQN_H_
#define _DQN_H_

#include "dqn_interface.h"


#define DQN_SARSA           ((unsigned int)0)
#define DQN_Q_LEARNING      ((unsigned int)1)

class DQN: public DQNInterface
{
  public:
    DQN();
    DQN(  Json::Value &json_config,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size,
          bool normalise);

    DQN(  std::string json_config_file_name,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size,
          bool normalise);

    DQN( std::string json_config_file_name,
         sGeometry state_geometry,
         unsigned int actions_count);

    DQN( std::string json_config_file_name);

    virtual ~DQN();

    void init(  Json::Value &json_config,
                float gamma,
                sGeometry state_geometry,
                unsigned int actions_count,
                unsigned int experience_buffer_size,
                bool normalise);


    void compute_q_values(std::vector<float> &state);

    void learn();
    void test();

    void new_batch();

    bool is_full();

    private:
        unsigned int algorithm;
};

#endif
