#include <iostream>
#include <math.h>
#include <sys/time.h>

using namespace std;

double tick(){
    timeval time;
    // nullptr是C++11的特性，gcc版本要大于5
    gettimeofday(&time, NULL);
    return time.tv_sec + time.tv_usec*1E-6;
}

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  // <<: 移位运算符
  int N = 1<<24; // 10M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  double t1 = tick();
  // Run kernel on 10M elements on the CPU
  add(N, x, y);
  double t2 = tick();

  cout << "Completed in: " << t2 - t1 << " seconds" << endl;

  // Check for errors (all values should be 3.0f)
  // float maxError = 0.0f;
  // for (int i = 0; i < N; i++)
  //   maxError = fmax(maxError, fabs(y[i]-3.0f));
  // cout << "Max error: " << maxError << endl;

  // Free memory
  delete [] x;
  delete [] y;

  return 0;
}