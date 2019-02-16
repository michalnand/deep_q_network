#ifndef _DQNP_H_
#define _DQNP_H_

#include "dqn_interface.h"


class DQNP: public DQNInterface
{
  protected:
    std::vector<float> priority;

  public:
    DQNP();
    DQNP(  Json::Value &json_config,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size,
          bool normalise);

    DQNP( std::string json_config_file_name,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size,
          bool normalise);

    DQNP( std::string json_config_file_name,
          sGeometry state_geometry,
          unsigned int actions_count);

    DQNP( std::string json_config_file_name);


    virtual ~DQNP();

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
};

#endif
