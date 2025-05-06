#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

class Graph
{
    int noV;
    int noE;
    vector<vector<int>> g;

public:
    Graph(int noV)
    {
        this->noV = noV;
        for (int i = 0; i <= noV - 1; i++)
        {
            g.push_back(vector<int>(noV, 0));
        }
    }

    void addEdges(int u, int v)
    {
        g[u][v] = 1;
        g[v][u] = 1;
    }

    void sequentialBFS(int source)
    {
        queue<int> q;
        vector<bool> isV(noV, false);
        q.push(source);
        isV[source] = true;
        while (!q.empty())
        {
            // #pragma omp parallel shared( q , isV )
            // {

            // }

            // pragma omp single
            int curr = q.front();
            // todo: memorize : front
            print(curr);
            q.pop();

            // pragma omp for
            for (int i = 0; i < noV; i++)
            {
                if (g[curr][i] == 1 && isV[i] == false)
                {
                    // pragma omp critical{}
                    q.push(i);
                    isV[i] = true;
                }
            }
        }
    }

    void parallelBFS(int source)
    {
        queue<int> q;
        vector<bool> isV(noV, false);
        q.push(source);
        isV[source] = true;
        while (!q.empty())
        {
            int curr;
#pragma omp parallel shared(q, isV)
            {
#pragma omp single
                {
                    curr = q.front();
                    // todo: memorize : front
                    print(curr);
                    q.pop();
                }

#pragma omp for
                for (int i = 0; i < noV; i++)
                {
                    if (g[curr][i] == 1 && isV[i] == false)
                    {

                        {
                            q.push(i);
                            isV[i] = true;
                        }
                    }
                }
            }
        }
    }

    void sequentialDFS(int source)
    {
        vector<bool> isV(noV, false);
        sDFSutil(isV, source);
    }

    void sDFSutil(vector<bool> &isV, int curr)
    {
        print(curr);
        isV[curr] = true;
        for (int i = 0; i < noV; i++)
        {
            if (g[curr][i] == 1)
            {
                {
                    if (isV[i] == false)
                        sDFSutil(isV, i);
                }
            }
        }
    }

    void parallelDFS(int source)
    {
        vector<bool> isV(noV, false);
        pDFSutil(isV, source);
    }

    void pDFSutil(vector<bool> isV, int curr)
    {
        print(curr);
        isV[curr] = true;

#pragma omp for
        for (int i = 0; i < noV; i++)
        {
            if (g[curr][i] == 1)
            {
#pragma omp critical
                {
                    if (isV[i] == false)
                        sDFSutil(isV, i);
                }
            }
        }
    }

    void print(int val)
    {
        cout << val << " ";
    }
};

int main()
{
    Graph graph1(5);
    graph1.addEdges(0, 1);
    graph1.addEdges(0, 2);
    graph1.addEdges(0, 4);
    graph1.addEdges(1, 3);
    graph1.addEdges(2, 3);

    // int v = 5 ;
    // Graph graph1(v) ;
    // for(int i = 0 ; i < v * 5 ; i++){
    //     graph1.addEdges( rand()% v , rand()% v ) ;
    // }

    // case1 ;
    auto start = high_resolution_clock ::now();
    graph1.sequentialBFS(0);
    auto end = high_resolution_clock ::now();
    cout << endl
         << duration_cast<microseconds>(end - start).count() << endl;

    // case 2
    start = high_resolution_clock ::now();
    graph1.parallelBFS(0);
    end = high_resolution_clock ::now();
    cout << endl
         << duration_cast<microseconds>(end - start).count() << endl;

    // case 3
    start = high_resolution_clock ::now();
    graph1.sequentialDFS(0);
    end = high_resolution_clock ::now();
    cout << endl
         << duration_cast<microseconds>(end - start).count() << endl;

    // case 4
    start = high_resolution_clock ::now();
    graph1.parallelDFS(0);
    end = high_resolution_clock ::now();
    cout << endl
         << duration_cast<microseconds>(end - start).count() << endl;

    return 0;
}
