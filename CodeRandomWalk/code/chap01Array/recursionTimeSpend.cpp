#include<iostream>

using namespace std;
// long pow(int x, int y);
long forLoop(int x, int y){
	int res = 1;
	for(int i=0; i<y; i++){
		res = res * x;
	}
	return res; // O(n)
}

// what if we use recursion ?
long recNoOptimize(int x, int y){
	if(y == 1)return x;
	else if(y%2 == 1)return recNoOptimize(x, y/2)*recNoOptimize(x, y/2)*x;
	else return recNoOptimize(x, y/2)*recNoOptimize(x, y/2);
} // actually if we draw the recursion tree, the time still O(n)

long recOptimize(int x, int y){
	if(y == 1)return x;
	else {
		int t = recOptimize(x, y/2);
		return y%2==1? t*t*x : t*t;
	}
}

long fib3(int f, int s, int n){
    if(n==1 || n==2)return 1;
    if(n==3)return f+s;
    return fib3(s, f+s, n-1);
}

int main() {
	cout << forLoop(2, 9) << endl;	
	cout << recNoOptimize(2, 9) << endl;	
	cout << recOptimize(2, 9) << endl;
	cout << fib3(1, 1, 9);
	// 1 1 2 3 5 8 13 21
}