#ifndef _DQNP_H_
#define _DQNP_H_

#include "dqn_interface.h"

struct sDQNPExperienceBuffer
{
  std::vector<float> state;
  std::vector<float> q_values;
  unsigned int action;

  float reward;

  bool is_final;
};

class DQNP: public DQNInterface
{
  protected:
    unsigned int current_ptr;
    std::vector<float> priority;
    std::vector<sDQNPExperienceBuffer> experience_buffer;

    float gamma;

    CNN *cnn;

  public:
    DQNP();
    DQNP(  Json::Value &json_config,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size);

    virtual ~DQNP();

    void init(  Json::Value &json_config,
                float gamma,
                sGeometry state_geometry,
                unsigned int actions_count,
                unsigned int experience_buffer_size);


    void compute_q_values(std::vector<float> &state);
    void add(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float reward);
    void add_final(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float final_reward);

    void learn();
    void test();

    void new_batch();

    bool is_full();

    void print();


  protected:
    void buffer_clear();
};

#endif
