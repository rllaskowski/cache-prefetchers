#include <iostream>

int main() {
    // table
    int a[] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) {
        int sum = 0;
        for (int j = 0; j < 5; j++) {
            sum += a[j];
        }
        // std::cout << sum << std::endl;
    }
    return 0;
}
