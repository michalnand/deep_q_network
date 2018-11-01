#ifndef _RANDOM_DISTRIBUTION_H_
#define _RANDOM_DISTRIBUTION_H_

#include <vector>

class RandomDistribution
{
  private:
    std::vector<double> p;

  public:
    RandomDistribution();
    RandomDistribution(std::vector<float> &p, int count = -1);
    RandomDistribution(RandomDistribution& other);
    RandomDistribution(const RandomDistribution& other);

    virtual ~RandomDistribution();
    RandomDistribution& operator= (RandomDistribution& other);
    RandomDistribution& operator= (const RandomDistribution& other);

  protected:
    void copy(RandomDistribution& other);
    void copy(const RandomDistribution& other);

  public:
    void set(std::vector<float> &p_, int count = -1);
    unsigned int get();

  private:
    double rnd();
    unsigned int search(double value);
};

#endif
