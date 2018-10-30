#include "dqn_compare.h"

#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <math.h>




DQNCompare::DQNCompare()
{
  clear();
}

DQNCompare::DQNCompare(unsigned int output_size)
{
  set_output_size(output_size);
}

DQNCompare::~DQNCompare()
{

}


void DQNCompare::clear()
{
  output_size = 0;

  for (unsigned int i = 0; i < output_size; i++)
  {
    h_target[i].clear();
    h_resulted[i].clear();
    h_error[i].clear();
  }

  h_target.clear();
  h_resulted.clear();
  h_error.clear();

  h_target_summary.clear();
  h_resulted_summary.clear();
  h_error_summary.clear();
  h_error_squared.clear();

  error_average_squared = 0.0;
  error_min_squared     = 1000000000.0*output_size;
  error_max_squared     = -error_min_squared;

  count = 0;
}

void DQNCompare::set_output_size(unsigned int output_size)
{
  clear();

  this->output_size = output_size;

  h_target.resize(output_size);
  h_resulted.resize(output_size);
  h_error.resize(output_size);



  for (unsigned int i = 0; i < output_size; i++)
  {
    h_target[i].clear();
    h_resulted[i].clear();
    h_error[i].clear();
  }

  h_target_summary.clear();
  h_resulted_summary.clear();
  h_error_summary.clear();
  h_error_squared.clear();

  h_action.clear();
}

void DQNCompare::compare(std::vector<float> &target_value, std::vector<float> &output_value, unsigned int action)
{
  for (unsigned int i = 0; i < output_size; i++)
  {
    h_target[i].add(target_value[i]);
    h_resulted[i].add(output_value[i]);

    float error = target_value[i] - output_value[i];
    float error_squared = pow(error, 2.0);

    h_error[i].add(error);

    h_target_summary.add(target_value[i]);
    h_resulted_summary.add(output_value[i]);
    h_error_summary.add(error);
    h_error_squared.add(error_squared);

    error_average_squared+= error_squared;

    if (error_squared < error_min_squared)
      error_min_squared = error_squared;

    if (error_squared > error_max_squared)
      error_max_squared = error_squared;

    count++;
  }

  h_action.add(action);
}

void DQNCompare::process(int fixed_bars_count)
{
  if (count == 0)
    return;

  int bars_count;

  if (fixed_bars_count > 0)
  {
    bars_count = fixed_bars_count;
  }
  else
  {
    bars_count = count/100;

    if (bars_count > 500)
      bars_count = 500;

    if (bars_count < 50)
      bars_count = 50;

    if ((bars_count%2) == 0)
      bars_count+= 1;
  }


  for (unsigned int i = 0; i < output_size; i++)
  {
    h_target[i].compute(bars_count);
    h_resulted[i].compute(bars_count);
    h_error[i].compute(bars_count);
  }

  h_target_summary.compute(bars_count);
  h_resulted_summary.compute(bars_count);
  h_error_summary.compute(bars_count);
  h_error_squared.compute(bars_count);

  h_action.compute(output_size);

  error_average_squared = error_average_squared/count;

  json_result = process_json_result();
}

float DQNCompare::get_error_average_squared()
{
  return error_average_squared;
}

float DQNCompare::get_error_min_squared()
{
  return error_min_squared;
}

float DQNCompare::get_error_max_squared()
{
  return error_max_squared;
}


void DQNCompare::save_text_file(std::string log_file_name_prefix)
{
  for (unsigned int i = 0; i < output_size; i++)
  {
    h_target[i].save(log_file_name_prefix + "full/" + "q_target_" + std::to_string(i) + ".log");
    h_resulted[i].save(log_file_name_prefix + "full/" + "q_resulted_" + std::to_string(i) + ".log");
    h_error[i].save(log_file_name_prefix + "full/" + "q_error_" + std::to_string(i) + ".log");
  }

  h_target_summary.save(log_file_name_prefix + "q_target_summary" + ".log");
  h_resulted_summary.save(log_file_name_prefix + "q_resulted_summary" + ".log");
  h_error_summary.save(log_file_name_prefix + "q_error_summary" + ".log");
  h_error_squared.save(log_file_name_prefix + "q_error_squared" + ".log");

  h_action.save(log_file_name_prefix + "q_action" + ".log");
}

void DQNCompare::save_json_file(std::string json_file_name)
{
  JsonConfig json;
  json.result = json_result;

  json.save(json_file_name);
}

Json::Value DQNCompare::process_json_result()
{
  Json::Value result;

  result["summary"]["count"]          = count;
  result["summary"]["error_average"]  = error_average_squared;
  result["summary"]["error_min"]      = error_min_squared;
  result["summary"]["error_max"]      = error_max_squared;

  for (unsigned int i = 0; i < h_target_summary.get_count(); i++)
  {
    result["summary"]["q_target_summary"][i]["value"] = h_target_summary.get(i).value;
    result["summary"]["q_target_summary"][i]["count"] = h_target_summary.get(i).count;
    result["summary"]["q_target_summary"][i]["normalised_count"] = h_target_summary.get(i).normalised_count;
  } 

  for (unsigned int i = 0; i < h_resulted_summary.get_count(); i++)
  {
    result["summary"]["q_resulted_summary"][i]["value"] = h_resulted_summary.get(i).value;
    result["summary"]["q_resulted_summary"][i]["count"] = h_resulted_summary.get(i).count;
    result["summary"]["q_resulted_summary"][i]["normalised_count"] = h_resulted_summary.get(i).normalised_count;
  }

  for (unsigned int i = 0; i < h_error_summary.get_count(); i++)
  {
    result["summary"]["q_error_summary"][i]["value"] = h_error_summary.get(i).value;
    result["summary"]["q_error_summary"][i]["count"] = h_error_summary.get(i).count;
    result["summary"]["q_error_summary"][i]["normalised_count"] = h_error_summary.get(i).normalised_count;
  }

  for (unsigned int i = 0; i < h_error_squared.get_count(); i++)
  {
    result["summary"]["q_error_squared"][i]["value"] = h_error_squared.get(i).value;
    result["summary"]["q_error_squared"][i]["count"] = h_error_squared.get(i).count;
    result["summary"]["q_error_squared"][i]["normalised_count"] = h_error_squared.get(i).normalised_count;
  }

  for (unsigned int j = 0; j < output_size; j++)
  for (unsigned int i = 0; i < h_error_summary.get_count(); i++)
  {
    result["detailed"][j]["q_target"][i]["value"] = h_target[j].get(i).value;
    result["detailed"][j]["q_target"][i]["count"] = h_target[j].get(i).count;
    result["detailed"][j]["q_target"][i]["normalised_count"] = h_target[j].get(i).normalised_count;

    result["detailed"][j]["q_resulted"][i]["value"] = h_resulted[j].get(i).value;
    result["detailed"][j]["q_resulted"][i]["count"] = h_resulted[j].get(i).count;
    result["detailed"][j]["q_resulted"][i]["normalised_count"] = h_resulted[j].get(i).normalised_count;

    result["detailed"][j]["q_error"][i]["value"] = h_error[j].get(i).value;
    result["detailed"][j]["q_error"][i]["count"] = h_error[j].get(i).count;
    result["detailed"][j]["q_error"][i]["normalised_count"] = h_error[j].get(i).normalised_count;
  }

  for (unsigned int j = 0; j < output_size; j++)
  {
    result["actions"][j]["value"] = h_action.get(j).value;
    result["actions"][j]["count"] = h_action.get(j).count;
    result["actions"][j]["normalised_count"] = h_action.get(j).normalised_count;
  }

  return result;
}
