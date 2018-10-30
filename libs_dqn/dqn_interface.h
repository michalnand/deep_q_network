#ifndef _DQN_INTERFACE_H_
#define _DQN_INTERFACE_H_

#include <cnn.h>
#include <dqn_compare.h>

class DQNInterface
{
  protected:
    sGeometry state_geometry;
    unsigned int state_size;
    unsigned int actions_count;
    unsigned int experience_buffer_size;

    std::vector<float> q_values;

    DQNCompare compare;

  public:
    DQNInterface();

    DQNInterface(sGeometry state_geometry, unsigned int actions_count, unsigned int experience_buffer_size);
    virtual ~DQNInterface();

    void init_interface(sGeometry state_geometry, unsigned int actions_count, unsigned int experience_buffer_size);

    std::vector<float>& get_q_values();
    float get_max_q_value();

    virtual void compute_q_values(std::vector<float> &state);
    virtual void add(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float reward);
    virtual void add_final(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float final_reward);
    virtual void learn();
    virtual void new_batch();

    virtual bool is_full();

    virtual void test();
    DQNCompare& get_compare_result();


  protected:
    float saturate(float value, float min, float max);
    unsigned int argmax(std::vector<float> &v);
};

#endif
