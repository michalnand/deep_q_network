#ifndef _DQN_INTERFACE_H_
#define _DQN_INTERFACE_H_

#include <cnn.h>
#include <dqn_compare.h>


struct sDQNExperienceBuffer
{
  std::vector<float> state;
  std::vector<float> q_values;
  unsigned int action;

  float reward;

  bool is_final;
};

class DQNInterface
{
  protected:
    sGeometry state_geometry;
    unsigned int state_size;
    unsigned int actions_count;
    unsigned int experience_buffer_size;
    unsigned int current_ptr;

    float gamma;

    std::vector<sDQNExperienceBuffer> experience_buffer;


    std::vector<float> q_values;

    DQNCompare compare;

    CNN *cnn;

  public:
    DQNInterface();

    DQNInterface(sGeometry state_geometry, unsigned int actions_count, unsigned int experience_buffer_size);
    virtual ~DQNInterface();

    void init_interface(sGeometry state_geometry, unsigned int actions_count, unsigned int experience_buffer_size);
    void buffer_clear();

    std::vector<float>& get_q_values();
    float get_max_q_value();

    void add(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float reward);
    void add_final(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float final_reward);


    virtual void compute_q_values(std::vector<float> &state);
    virtual void learn();
    virtual void new_batch();

    virtual bool is_full();

    virtual void test();
    DQNCompare& get_compare_result();


    virtual void save(std::string file_name_prefix);
    void load_weights(std::string file_name_prefix);

    virtual void print();


  protected:
    float saturate(float value, float min, float max);
    unsigned int argmax(std::vector<float> &v);
};

#endif
