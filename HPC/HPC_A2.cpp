#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

// Parallel Bubble Sort
void parallelBubbleSort(vector<int> &array)
{
    int size = array.size();
    for (int k = 0; k < size; k++)
    {
        if (k % 2 == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < size - 1; i += 2)
            {
                if (array[i] > array[i + 1])
                {
                    swap(array[i], array[i + 1]);
                }
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 1; i < size - 1; i += 2)
            {
                if (array[i] > array[i + 1])
                {
                    swap(array[i], array[i + 1]);
                }
            }
        }
    }
}

// Standard Bubble Sort
void bubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
void merge(vector<int> &array, int low, int mid, int high)
{
    int size = array.size();
    int temp[size];
    int i = low;
    int j = mid + 1;
    int k = 0;
    while ((i <= mid) && (j <= high))
    {
        if (array[i] >= array[j])
        {
            temp[k] = array[j];
            k++;
            j++;
        }
        else
        {
            temp[k] = array[i];
            k++;
            i++;
        }
    }
    while (i <= mid)
    {
        temp[k] = array[i];
        k++;
        i++;
    }
    while (j <= high)
    {
        temp[k] = array[j];
        k++;
        j++;
    }

    k = 0;
    for (int i = low; i <= high; i++)
    {
        array[i] = temp[k];
        k++;
    }
}

void mergeSort(vector<int> &arr, int l, int r)
{
    // Merge sort function
    if (l < r)
    {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void parallelMergeSort(vector<int> &arr, int low, int high)
{
    // Parallel merge sort function
    if (low < high)
    {
        int mid = low + (high - low) / 2;
        if ((high - low) < 1000)
        { // Threshold to prevent overhead on small arrays
            mergeSort(arr, low, high);
            return;
        }
#pragma omp parallel
        {
#pragma omp single nowait
            {
#pragma omp task
                parallelMergeSort(arr, low, mid);
#pragma omp task
                parallelMergeSort(arr, mid + 1, high);
#pragma omp taskwait
                merge(arr, low, mid, high);
            }
        }
    }
}

int main()
{
    const int size = 10000; // Size of the array
    vector<int> arr(size), arr1(size), arr2(size), arr3(size);

    // Initialize array with random values
    for (int i = 0; i < size; ++i)
    {
        arr[i] = rand() % 1000;
        arr1[i] = arr[i];
        arr2[i] = arr[i];
        arr3[i] = arr[i];
    }

    // Sequential bubble sort
    auto start = high_resolution_clock::now();
    bubbleSort(arr);
    auto stop = high_resolution_clock::now();
    auto seq_time = duration_cast<milliseconds>(stop - start);
    cout << "Time taken by sequential bubble sort: " << seq_time.count() << " ms" << endl;

    // Parallel bubble sort
    start = high_resolution_clock::now();
    parallelBubbleSort(arr1);
    stop = high_resolution_clock::now();
    auto par_time = duration_cast<milliseconds>(stop - start);
    cout << "Time taken by parallel bubble sort: " << par_time.count() << " ms" << endl;

    start = high_resolution_clock::now();
    mergeSort(arr2, 0, size - 1);
    stop = high_resolution_clock::now();
    auto seq_time_merge = duration_cast<milliseconds>(stop - start);
    cout << "Time taken by sequential merge sort: " << seq_time_merge.count() << " ms" << endl;

    start = high_resolution_clock::now();
    parallelMergeSort(arr3, 0, size - 1);
    stop = high_resolution_clock::now();
    auto par_time_merge = duration_cast<milliseconds>(stop - start);
    cout << "Time taken by parallel merge sort: " << par_time_merge.count() << " ms" << endl;

    return 0;
}