#ifndef _DQN_COMPARE_H_
#define _DQN_COMPARE_H_

#include <string>
#include <vector>

#include "json_config.h"
#include "histogram.h"

class DQNCompare
{
  private:
    unsigned int output_size;

    std::vector<Histogram> h_target;
    std::vector<Histogram> h_resulted;
    std::vector<Histogram> h_error;


    Histogram h_target_summary;
    Histogram h_resulted_summary;
    Histogram h_error_summary;
    Histogram h_error_squared;

    Histogram h_action;

    unsigned int count;

    float error_average_squared ;
    float error_min_squared     ;
    float error_max_squared     ;

    Json::Value json_result;

  public:
    DQNCompare();
    DQNCompare(unsigned int output_size);

    virtual ~DQNCompare();

    void clear();
    void set_output_size(unsigned int output_size);
 
    void compare(std::vector<float> &target_value, std::vector<float> &output_value, unsigned int action);
    void process(int fixed_bars_count = -1);

    float get_error_average_squared();
    float get_error_min_squared();
    float get_error_max_squared();


    void save_text_file(std::string log_file_name_prefix);
    void save_json_file(std::string json_file_name);
    Json::Value process_json_result();
};


#endif
