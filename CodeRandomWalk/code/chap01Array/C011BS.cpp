#include<iostream>
#include<vector>
using namespace std;

int midSearch(vector<int>& nums, int target){
	int left = 0;
	int right = nums.size() - 1;
	while (left <= right){
		int mid = left + (right-left)/2;
		// cout << left << " " << mid << " " << right << endl;
		if (nums[mid] < target)
			left = mid + 1;  // [mid+1, right]
		else if (nums[mid] > target)
			right = mid - 1; // [left, mid-1]
		else return mid;
	}
	return -1; // 找不到
}

int midS2(vector<int>& nums, int target){
	// [left, right)
	int left = 0;
	int right = nums.size();
	while (left < right){
		int mid = left + (right - left)/2;
		if (target > nums[mid]){
			left = mid + 1; // [mid+1, right)
		}else if(target < nums[mid]){
			right = mid; // [left, mid)
		}else{
			return mid;
		}
	}
	return -1;
}

// 递归版本
int msr01(int l, int r, vector<int>& nums, int target){
	if(l <= r){
		// 仅当l<=r时[l, r]内才会有元素
		int m = l + (r-l)/2;
		if(nums[m] > target){
			return msr01(l, m-1, nums, target);
		}else if(nums[m] < target){
			return msr01(m+1, r, nums, target);
		}else return m;
	}
	return -1;
}

int midSearchRec(vector<int>& nums, int target){
	int left = 0, right = nums.size()-1;
	return msr01(left, right, nums, target); // [left, right]
}


int main(){
	int x;
	cin >> x;
	vector<int> nums = {2, 5, 6, 9, 11, 12, 20};
	int res = midSearchRec(nums, x);
	cout << "a[" << res << "] = " << x << endl;
	return 0;
}