#ifndef _DDQN_H_
#define _DDQN_H_

#include "dqn_interface.h"

struct sDDQNExperienceBuffer
{
  std::vector<float> state;
  std::vector<float> q_values;
  unsigned int action;

  float reward;

  bool is_final;
};

class DDQN: public DQNInterface
{
  protected:
    unsigned int current_ptr;
    std::vector<float> priority;
    std::vector<sDDQNExperienceBuffer> experience_buffer;

    float gamma;

    CNN *cnn;

    std::vector<float> nn_output;

  public:
    DDQN();
    DDQN(  Json::Value &json_config,
          float gamma,
          sGeometry state_geometry,
          unsigned int actions_count,
          unsigned int experience_buffer_size);

    virtual ~DDQN();

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
    void q_values_to_nn_output(std::vector<float> &nn_output, std::vector<float> &q_values);
    void nn_output_to_q_values(std::vector<float> &q_values, std::vector<float> &nn_output);
    float v_average(std::vector<float> &v, int size = -1);

};

#endif
