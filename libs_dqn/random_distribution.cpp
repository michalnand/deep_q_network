#include "random_distribution.h"
#include <iostream>


RandomDistribution::RandomDistribution()
{

}

RandomDistribution::RandomDistribution(std::vector<float> &p, int count)
{
  set(p, count);
}

RandomDistribution::RandomDistribution(RandomDistribution& other)
{
  copy(other);
}

RandomDistribution::RandomDistribution(const RandomDistribution& other)
{
  copy(other);
}

RandomDistribution::~RandomDistribution()
{

}

RandomDistribution& RandomDistribution::operator= (RandomDistribution& other)
{
  copy(other);

  return *this;
}

RandomDistribution& RandomDistribution::operator= (const RandomDistribution& other)
{
  copy(other);

  return *this;
}

void RandomDistribution::copy(RandomDistribution& other)
{
  p = other.p;
}

void RandomDistribution::copy(const RandomDistribution& other)
{
  p = other.p;
}

void RandomDistribution::set(std::vector<float> &p_, int count)
{
  if (count < 0)
    count = p_.size();
 
  p.resize(count);
 

  for (unsigned int i = 0; i < (unsigned int)count; i++)
  {
    double v = p_[i];

    if (v < 0.0)
      v = -v;

    p[i] = v;
  }

  double sum;

  sum = 0.0;
  for (unsigned int i = 0; i < p.size(); i++)
    sum+= p[i];

  if (sum > 0.0)
  {
    for (unsigned int i = 0; i < p.size(); i++)
      p[i]/= sum;
  }
 

 
  sum = 0.0;
  for (unsigned int i = 0; i < p.size(); i++)
  {
    sum+= p[i];
    p[i] = sum;
  } 
}

unsigned int RandomDistribution::get()
{
  double v = rnd();

  return search(v);
}


double RandomDistribution::rnd()
{ 
  return ((double) rand() / (RAND_MAX)) ; 
} 

unsigned int RandomDistribution::search(double value)
{
  unsigned int center = 0;
  unsigned int left   = 0;
  unsigned int right  = p.size()-1;

  unsigned int result = 0; 

  if (value > p[0])  
  {
    while (left <= right)
    { 
      center = (left + right)/2;
  
      if (value < p[center])
        right = center-1;
      else
        left  = center+1;
    } 

    result = 1 + (right + left)/2;
  }

   
  return result;
}
