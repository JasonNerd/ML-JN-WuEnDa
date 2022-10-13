#include<iostream>
#include<chrono>
#include<thread>
using namespace std;
using namespace chrono;

void fN(long long n){
    long long i=0;
    while(i++<n);
}

void fN2(long long n){
	long long k=0;
	for(long long i=0; i<n; i++){
		for(long long j=0; j<n; j++){
			++k;
		}
	}
}

void fNlgN(long long n){
	long long k=0;
	for(long long i=0; i<n; i++){
		for(long long j=1; j<n; j *= 2){
			++k;
		}
	}
}

int msCatch(long long n, void (*func)(long long n)){
	milliseconds startTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	func(n);
	milliseconds endTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	return milliseconds(endTime).count() - milliseconds(startTime).count();	
}

int main(){
	int n, choice, t;
	while(true){
		cin >> n >> choice;
		if(n<=0 || choice<=0)break;
		if(choice==1)
			t = msCatch(n, fN); // 2e9
		else if(choice==2)
			t = msCatch(n, fN2); // 42500 约 4e4
		else
			t = msCatch(n, fNlgN); // 6e7
		cout << "time: " << t << endl;
	}
    return 0;
}