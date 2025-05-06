#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int minvalp(vector<int> &arr)
{
    int minval = arr[0];
#pragma omp parallel for reduction(min : minval)
    for (int i = 1; i < arr.size(); i++)
    {
        minval = min(arr[i], minval);
    }
    return minval;
}

int minvals(vector<int> &arr)
{
    int minval = arr[0];
    for (int i = 1; i < arr.size(); i++)
    {
        minval = min(arr[i], minval);
    }
    return minval;
}

int maxvalp(vector<int> &arr)
{
    int maxval = arr[0];
#pragma omp parallel for reduction(max : maxval)
    for (int i = 1; i < arr.size(); i++)
    {
        maxval = max(arr[i], maxval);
    }
    return maxval;
}

int maxvals(vector<int> &arr)
{
    int maxval = arr[0];
    for (int i = 1; i < arr.size(); i++)
    {
        maxval = max(arr[i], maxval);
    }
    return maxval;
}

int sump(vector<int> &arr)
{
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < arr.size(); i++)
    {
        sum += arr[i];
    }
    return sum;
}

int sums(vector<int> &arr)
{
    int sum = 0;
    for (int i = 0; i < arr.size(); i++)
    {
        sum += arr[i];
    }
    return sum;
}

int avgp(vector<int> &arr)
{
    return sump(arr) / arr.size();
}

int avgs(vector<int> &arr)
{
    return sums(arr) / arr.size();
}

int main()
{
    int n = 1000000; // 1 million elements
    vector<int> arr(n);

    srand(time(0));
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 10000; // random numbers between 0 and 9999
    }

    cout << "Array size: " << n << endl;
    cout << "OpenMP threads: " << omp_get_max_threads() << endl
         << endl;

    double starttime, endtime;

    // Parallel Minimum
    starttime = omp_get_wtime();
    cout << "Parallel Min: " << minvalp(arr) << endl;
    endtime = omp_get_wtime();
    cout << "Parallel Min Time: " << endtime - starttime << " seconds" << endl;

    // Sequential Minimum
    starttime = omp_get_wtime();
    cout << "Sequential Min: " << minvals(arr) << endl;
    endtime = omp_get_wtime();
    cout << "Sequential Min Time: " << endtime - starttime << " seconds" << endl;

    // Parallel Maximum
    starttime = omp_get_wtime();
    cout << "Parallel Max: " << maxvalp(arr) << endl;
    endtime = omp_get_wtime();
    cout << "Parallel Max Time: " << endtime - starttime << " seconds" << endl;

    // Sequential Maximum
    starttime = omp_get_wtime();
    cout << "Sequential Max: " << maxvals(arr) << endl;
    endtime = omp_get_wtime();
    cout << "Sequential Max Time: " << endtime - starttime << " seconds" << endl;

    // Parallel Sum
    starttime = omp_get_wtime();
    cout << "Parallel Sum: " << sump(arr) << endl;
    endtime = omp_get_wtime();
    cout << "Parallel Sum Time: " << endtime - starttime << " seconds" << endl;

    // Sequential Sum
    starttime = omp_get_wtime();
    cout << "Sequential Sum: " << sums(arr) << endl;
    endtime = omp_get_wtime();
    cout << "Sequential Sum Time: " << endtime - starttime << " seconds" << endl;

    // Parallel Average
    starttime = omp_get_wtime();
    cout << "Parallel Avg: " << avgp(arr) << endl;
    endtime = omp_get_wtime();
    cout << "Parallel Avg Time: " << endtime - starttime << " seconds" << endl;

    // Sequential Average
    starttime = omp_get_wtime();
    cout << "Sequential Avg: " << avgs(arr) << endl;
    endtime = omp_get_wtime();
    cout << "Sequential Avg Time: " << endtime - starttime << " seconds" << endl;

    return 0;
}